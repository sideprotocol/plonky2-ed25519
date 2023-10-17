#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use anyhow::Result;
use clap::Parser;
use core::num::ParseIntError;
use std::fs;
use log::{info, Level, LevelFilter};
use plonky2::gates::noop::NoopGate;
use plonky2::hash::hash_types::{HashOut, RichField};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{
    CircuitConfig, CommonCircuitData, VerifierCircuitTarget, VerifierOnlyCircuitData,
};
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig, Hasher, PoseidonGoldilocksConfig};
use plonky2::plonk::proof::{CompressedProofWithPublicInputs, ProofWithPublicInputs};
use plonky2::plonk::prover::{my_prove, prove};
use plonky2::recursion::tree_recursion::{
    check_tree_proof_verifier_data, common_data_for_recursion, set_tree_recursion_leaf_data_target,
    TreeRecursionLeafData,
};
use plonky2::util::timing::TimingTree;
use plonky2_ed25519::curve::eddsa::{
    SAMPLE_MSG1, SAMPLE_MSG2, SAMPLE_PK1, SAMPLE_SIG1, SAMPLE_SIG2,
};
use plonky2_ed25519::gadgets::eddsa::{fill_circuits, make_verify_circuits};
use plonky2_field::extension::Extendable;
use plonky2_field::goldilocks_field::GoldilocksField;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use std::process::exit;
use plonky2::timed;
use plonky2_cuda;

use serde::{Serialize, Deserialize};
use serde_json;
use std::io::prelude::*;
use plonky2::fri::oracle::CudaInnerContext;
use plonky2::hash::hashing::{hash_n_to_hash_no_pad, hash_n_to_m_no_pad, PlonkyPermutation};
use plonky2::hash::poseidon::{Poseidon, PoseidonHash, PoseidonPermutation};
use plonky2_field::fft::fft_root_table;
use plonky2_field::types::Field;
use plonky2_field::zero_poly_coset::ZeroPolyOnCoset;
use plonky2_util::{log2_ceil, log2_strict};
use rustacuda::memory::{cuda_malloc, DeviceBox};
use rustacuda::prelude::*;
use rustacuda::memory::DeviceBuffer;

// #[macro_use]
// extern crate rustacuda;
// extern crate rustacuda_core;

// extern crate cuda;
// use cuda::runtime::{CudaError, cudaMalloc, cudaMemcpy, cudaFree};
// use cuda::runtime::raw::{cudaError_t, cudaError_enum};

type ProofTuple<F, C, const D: usize> = (
    ProofWithPublicInputs<F, C, D>,
    VerifierOnlyCircuitData<C, D>,
    CommonCircuitData<F, D>,
);

fn prove_ed25519<F: RichField + Extendable<D> + rustacuda::memory::DeviceCopy, C: GenericConfig<D, F = F>, const D: usize>(
    msg: &[u8],
    sigv: &[u8],
    pkv: &[u8],
) -> Result<ProofTuple<F, C, D>>
where
    [(); C::Hasher::HASH_SIZE]:,
{
    let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::wide_ecc_config());

    let targets = make_verify_circuits(&mut builder, msg.len());
    let mut pw = PartialWitness::new();
    fill_circuits::<F, D>(&mut pw, msg, sigv, pkv, &targets);

    println!(
        "Constructing inner proof with {} gates",
        builder.num_gates()
    );
    println!("index num: {}", builder.virtual_target_index);

    let data = {
        if
        Path::new("sigma_vecs.bin").exists() &&
            Path::new("constants_sigmas_commitment.polynomials.bin").exists() &&
            Path::new("constants_sigmas_commitment.leaves.bin").exists() &&
            Path::new("constants_sigmas_commitment.digests.bin").exists() &&
            Path::new("constants_sigmas_commitment.caps.bin").exists() &&
            Path::new("forest.bin").exists() {
            builder.my_build::<C>()
        } else {
            builder.build::<C>()
        }
    };
    println!("gates: {}", data.common.gates.len());
    // {
    //     let proof = ProofWithPublicInputs::from_bytes(fs::read("ed25519.proof").expect("无法读取文件"), &data.common)?;
    //     let timing = TimingTree::new("verify", Level::Info);
    //     data.verify(proof.clone()).expect("verify error");
    //     timing.print();
    //     exit(0);
    // }

    let mut ctx;
    {
        rustacuda::init(CudaFlags::empty()).unwrap();
        let device_index = 0;
        let device = rustacuda::prelude::Device::get_device(device_index).unwrap();
        let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device).unwrap();
        let stream  = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let stream2 = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let poly_num: usize = 234;
        let values_num_per_poly  = 1<<18;
        let blinding = false;
        const SALT_SIZE: usize = 4;
        let rate_bits = 3;
        let cap_height = 4;

        let lg_n = log2_strict(values_num_per_poly );
        let n_inv = F::inverse_2exp(lg_n);
        let n_inv_ptr  : *const F = &n_inv;

        let fft_root_table_max = fft_root_table(1<<(lg_n + rate_bits)).concat();
        let fft_root_table_deg    = fft_root_table(1 << lg_n).concat();


        let salt_size = if blinding { SALT_SIZE } else { 0 };
        let values_flatten_len = poly_num*values_num_per_poly;
        let ext_values_flatten_len = (values_flatten_len+salt_size*values_num_per_poly) * (1<<rate_bits);
        let mut ext_values_flatten :Vec<F> = Vec::with_capacity(ext_values_flatten_len);
        unsafe {
            ext_values_flatten.set_len(ext_values_flatten_len);
        }

        let mut values_flatten :Vec<F> = Vec::with_capacity(values_flatten_len);
        unsafe {
            values_flatten.set_len(values_flatten_len);
        }

        let (values_flatten2, ext_values_flatten2) = {
            let poly_num = 20;
            let values_flatten_len = poly_num*values_num_per_poly;
            let ext_values_flatten_len = (values_flatten_len+salt_size*values_num_per_poly) * (1<<rate_bits);
            let mut ext_values_flatten :Vec<F> = Vec::with_capacity(ext_values_flatten_len);
            unsafe {
                ext_values_flatten.set_len(ext_values_flatten_len);
            }

            let mut values_flatten :Vec<F> = Vec::with_capacity(values_flatten_len);
            unsafe {
                values_flatten.set_len(values_flatten_len);
            }
            (values_flatten, ext_values_flatten)
        };

        let (values_flatten3, ext_values_flatten3) = {
            let poly_num = data.common.config.num_challenges * (1<<rate_bits);
            let values_flatten_len = poly_num*values_num_per_poly;
            let ext_values_flatten_len = (values_flatten_len+salt_size*values_num_per_poly) * (1<<rate_bits);
            let mut ext_values_flatten :Vec<F> = Vec::with_capacity(ext_values_flatten_len);
            unsafe {
                ext_values_flatten.set_len(ext_values_flatten_len);
            }

            let mut values_flatten :Vec<F> = Vec::with_capacity(values_flatten_len);
            unsafe {
                values_flatten.set_len(values_flatten_len);
            }
            (values_flatten, ext_values_flatten)
        };

        let len_cap = (1 << cap_height);
        let num_digests = 2 * (values_num_per_poly*(1<<rate_bits) - len_cap);
        let num_digests_and_caps = num_digests + len_cap;
        let mut digests_and_caps_buf :Vec<<<C as GenericConfig<D>>::Hasher as Hasher<F>>::Hash> = Vec::with_capacity(num_digests_and_caps);
        unsafe {
            digests_and_caps_buf.set_len(num_digests_and_caps);
        }

        let digests_and_caps_buf2 = digests_and_caps_buf.clone();
        let digests_and_caps_buf3 = digests_and_caps_buf.clone();

        // let mut values_device = unsafe{
        //     DeviceBuffer::<F>::uninitialized(values_flatten_len)?
        // };

        let pad_extvalues_len = ext_values_flatten.len();
        // let mut ext_values_device = {
        //         let mut values_device = unsafe {
        //             DeviceBuffer::<F>::uninitialized(
        //             pad_extvalues_len
        //                 + ext_values_flatten_len
        //                 + digests_and_caps_buf.len()*4
        //             )
        //         }.unwrap();
        //
        //         values_device
	    // };

        let mut cache_mem_device = {
            let mut cache_mem_device = unsafe {
                DeviceBuffer::<F>::uninitialized(
                    values_flatten_len

                        + pad_extvalues_len
                        + ext_values_flatten_len

                        + digests_and_caps_buf.len()*4
                )
            }.unwrap();

            cache_mem_device
        };

        let root_table_device = {
                let mut root_table_device = DeviceBuffer::from_slice(&fft_root_table_deg).unwrap();
                root_table_device
	    };

        let root_table_device2 = {
                let mut root_table_device = DeviceBuffer::from_slice(&fft_root_table_max).unwrap();
                root_table_device
	    };

        let constants_sigmas_commitment_leaves_device =
            DeviceBuffer::from_slice(&data.prover_only.constants_sigmas_commitment.merkle_tree.leaves.concat()).unwrap();

        let shift_powers = F::coset_shift().powers().take(1<<(lg_n)).collect::<Vec<F>>();
        let shift_powers_device = {
                let mut shift_powers_device = DeviceBuffer::from_slice(&shift_powers).unwrap();
            shift_powers_device
	    };

        let shift_inv_powers = F::coset_shift().powers().take(1<<(lg_n+rate_bits)).map(|f| f.inverse()).collect::<Vec<F>>();
        let shift_inv_powers_device = {
            let mut shift_inv_powers_device = DeviceBuffer::from_slice(&shift_inv_powers).unwrap();
            shift_inv_powers_device
        };

        // unsafe
        // {
        //     let mut file = File::create("inv-powers.bin").unwrap();
        //     let v = shift_inv_powers;
        //     file.write_all(std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()*8));
        // }

        let quotient_degree_bits = log2_ceil(data.common.quotient_degree_factor);
        let points = F::two_adic_subgroup(data.common.degree_bits() + quotient_degree_bits);

        let z_h_on_coset = ZeroPolyOnCoset::new(data.common.degree_bits(), quotient_degree_bits);

        let points_device = DeviceBuffer::from_slice(&points).unwrap();
        let z_h_on_coset_evals_device = DeviceBuffer::from_slice(&z_h_on_coset.evals).unwrap();
        let z_h_on_coset_inverses_device = DeviceBuffer::from_slice(&z_h_on_coset.inverses).unwrap();
        let k_is_device = DeviceBuffer::from_slice(&data.common.k_is).unwrap();

        ctx = plonky2::fri::oracle::CudaInvContext{
            inner: CudaInnerContext{stream, stream2,},
            ext_values_flatten:   Arc::new(ext_values_flatten),
            values_flatten:       Arc::new(values_flatten),
            digests_and_caps_buf: Arc::new(digests_and_caps_buf),

            ext_values_flatten2:   Arc::new(ext_values_flatten2),
            values_flatten2:       Arc::new(values_flatten2),
            digests_and_caps_buf2: Arc::new(digests_and_caps_buf2),

            ext_values_flatten3:   Arc::new(ext_values_flatten3),
            values_flatten3:       Arc::new(values_flatten3),
            digests_and_caps_buf3: Arc::new(digests_and_caps_buf3),

            cache_mem_device,
            second_stage_offset: values_flatten_len + ext_values_flatten_len,
            root_table_device,
            root_table_device2,
            constants_sigmas_commitment_leaves_device,
            shift_powers_device,
            shift_inv_powers_device,

            points_device,
            z_h_on_coset_evals_device,
            z_h_on_coset_inverses_device,
            k_is_device,

            ctx: _ctx,
        };
    }


    for (i, gate) in data.common.gates.iter().enumerate() {
        println!("gate: {}", gate.0.id());
    }

    let mut timing = TimingTree::new("prove", Level::Debug);
    println!("num_gate_constraints: {}, num_constraints: {}, selectors_info: {:?}",
             data.common.num_gate_constraints, data.common.num_constants,
            data.common.selectors_info,
    );
    let proof = my_prove(
        &data.prover_only,
        &data.common,
        pw,
        &mut timing,
        &mut ctx,
    )?;

    timing.print();

    let timing = TimingTree::new("verify", Level::Info);
    data.verify(proof.clone()).expect("verify error");
    timing.print();
    let proof_bytes = proof.to_bytes();
    let mut file = File::create("ed25519.proof")?;
    file.write_all(&*proof_bytes)
        .expect("Leaf proof file write err");

    Ok((proof, data.verifier_only, data.common))
}

fn recursive_proof<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    InnerC: GenericConfig<D, F = F>,
    const D: usize,
>(
    inner1: &ProofTuple<F, InnerC, D>,
    inner2: Option<ProofTuple<F, InnerC, D>>,
    config: &CircuitConfig,
    min_degree_bits: Option<usize>,
) -> Result<ProofTuple<F, C, D>>
where
    InnerC::Hasher: AlgebraicHasher<F>,
    [(); C::Hasher::HASH_SIZE]:,
{
    let mut builder = CircuitBuilder::<F, D>::new(config.clone());
    let mut pw = PartialWitness::new();

    {
        let (inner_proof, inner_vd, inner_cd) = inner1;
        let pt = builder.add_virtual_proof_with_pis::<InnerC>(inner_cd);
        pw.set_proof_with_pis_target(&pt, inner_proof);
        builder.register_public_inputs(&*pt.public_inputs);

        let inner_data = VerifierCircuitTarget {
            constants_sigmas_cap: builder.add_virtual_cap(inner_cd.config.fri_config.cap_height),
            circuit_digest: builder.add_virtual_hash(),
        };
        pw.set_cap_target(
            &inner_data.constants_sigmas_cap,
            &inner_vd.constants_sigmas_cap,
        );
        pw.set_hash_target(inner_data.circuit_digest, inner_vd.circuit_digest);

        builder.verify_proof::<InnerC>(&pt, &inner_data, inner_cd);
    }

    if inner2.is_some() {
        let (inner_proof, inner_vd, inner_cd) = inner2.unwrap();
        let pt = builder.add_virtual_proof_with_pis::<InnerC>(&inner_cd);
        pw.set_proof_with_pis_target(&pt, &inner_proof);
        builder.register_public_inputs(&*pt.public_inputs);

        let inner_data = VerifierCircuitTarget {
            constants_sigmas_cap: builder.add_virtual_cap(inner_cd.config.fri_config.cap_height),
            circuit_digest: builder.add_virtual_hash(),
        };
        pw.set_hash_target(inner_data.circuit_digest, inner_vd.circuit_digest);
        pw.set_cap_target(
            &inner_data.constants_sigmas_cap,
            &inner_vd.constants_sigmas_cap,
        );

        builder.verify_proof::<InnerC>(&pt, &inner_data, &inner_cd);
    }
    builder.print_gate_counts(0);
    println!(
        "recur Constructing inner proof with {} gates",
        builder.num_gates()
    );

    if let Some(min_degree_bits) = min_degree_bits {
        // We don't want to pad all the way up to 2^min_degree_bits, as the builder will
        // add a few special gates afterward. So just pad to 2^(min_degree_bits
        // - 1) + 1. Then the builder will pad to the next power of two,
        // 2^min_degree_bits.
        let min_gates = (1 << (min_degree_bits - 1)) + 1;
        for _ in builder.num_gates()..min_gates {
            builder.add_gate(NoopGate, vec![]);
        }
    }

    let data = builder.build::<C>();

    let mut timing = TimingTree::new("prove", Level::Info);
    let proof = prove(&data.prover_only, &data.common, pw, &mut timing)?;
    timing.print();

    data.verify(proof.clone())?;

    test_serialization(&proof, &data.verifier_only, &data.common)?;
    Ok((proof, data.verifier_only, data.common))
}

fn benchmark() -> Result<()> {
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;
    let config = CircuitConfig::standard_recursion_config();

    let proof1 = prove_ed25519(
        SAMPLE_MSG1.as_bytes(),
        SAMPLE_SIG1.as_slice(),
        SAMPLE_PK1.as_slice(),
    )
    .expect("prove error 1");
    let proof2 = prove_ed25519(
        SAMPLE_MSG2.as_bytes(),
        SAMPLE_SIG2.as_slice(),
        SAMPLE_PK1.as_slice(),
    )
    .expect("prove error 2");

    // Recursively verify the proof
    let middle = recursive_proof::<F, C, C, D>(&proof1, Some(proof2), &config, None)?;
    let (_, _, cd) = &middle;
    info!(
        "Single recursion proof degree {} = 2^{}",
        cd.degree(),
        cd.degree_bits()
    );

    // Add a second layer of recursion to shrink the proof size further
    let outer = recursive_proof::<F, C, C, D>(&middle, None, &config, None)?;
    let (_, _, cd) = &outer;
    info!(
        "Double recursion proof degree {} = 2^{}",
        cd.degree(),
        cd.degree_bits()
    );

    Ok(())
}

/// Test serialization and print some size info.
fn test_serialization<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    proof: &ProofWithPublicInputs<F, C, D>,
    vd: &VerifierOnlyCircuitData<C, D>,
    cd: &CommonCircuitData<F, D>,
) -> Result<()>
where
    [(); C::Hasher::HASH_SIZE]:,
{
    let proof_bytes = proof.to_bytes();
    info!("Proof length: {} bytes", proof_bytes.len());
    let proof_from_bytes = ProofWithPublicInputs::from_bytes(proof_bytes, cd)?;
    assert_eq!(proof, &proof_from_bytes);

    let now = std::time::Instant::now();
    let compressed_proof = proof.clone().compress(&vd.circuit_digest, cd)?;
    let decompressed_compressed_proof = compressed_proof
        .clone()
        .decompress(&vd.circuit_digest, cd)?;
    info!("{:.4}s to compress proof", now.elapsed().as_secs_f64());
    assert_eq!(proof, &decompressed_compressed_proof);

    let compressed_proof_bytes = compressed_proof.to_bytes();
    info!(
        "Compressed proof length: {} bytes",
        compressed_proof_bytes.len()
    );
    let compressed_proof_from_bytes =
        CompressedProofWithPublicInputs::from_bytes(compressed_proof_bytes, cd)?;
    assert_eq!(compressed_proof, compressed_proof_from_bytes);

    Ok(())
}

#[derive(Parser)]
struct Cli {
    #[arg(short, long, default_value_t = 0)]
    benchmark: u8,
    #[arg(short, long, default_value = "./ed25519.proof")]
    output_path: PathBuf,
    #[arg(short, long)]
    msg: Option<String>,
    #[arg(short, long)]
    pk: Option<String>,
    #[arg(short, long)]
    sig: Option<String>,
}

pub fn decode_hex(s: &String) -> Result<Vec<u8>, ParseIntError> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
        .collect()
}

fn main() -> Result<()> {
    // Initialize logging
    let mut builder = env_logger::Builder::from_default_env();
    builder.format_timestamp(None);
    // builder.filter_level(LevelFilter::Debug);
    builder.try_init()?;

    let args = Cli::parse();
    if args.benchmark == 1 {
        // Run the benchmark
        benchmark()
    } else {
        if args.sig.is_none() || args.pk.is_none() || args.msg.is_none() {
            println!("The required arguments were not provided: --msg MSG_IN_HEX  --pk PUBLIC_KEY_IN_HEX  --sig SIGNATURE_IN_HEX");
            return Ok(());
        }

        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        // unsafe {
        //     let f = std::mem::transmute::<u64, F>(0xfffffffeffe00001);
        //     // let inv = std::mem::transmute::<u64, F>(0xbfa99fe2edeb56f5);
        //
        //     println!("inv: {:016X}", std::mem::transmute::<F, u64>(f.inverse()));
        //     // println!("n  : {:016X}", std::mem::transmute::<F, u64>(f));
        //     // println!("res: {:016X}", std::mem::transmute::<F, u64>(f * inv));
        //     //
        //     // fn split(x: u128) -> (u64, u64) {
        //     //     (x as u64, (x >> 64) as u64)
        //     // }
        //     //
        //     // println!("{:?}", split(f.0 as u128 * inv.0 as u128));
        // }

        // let inputs = [
        //     GoldilocksField(12057761340118092379),
        //     GoldilocksField(6921394802928742357),
        //     GoldilocksField(401572749463996457),
        //     GoldilocksField(8075242603528285606),
        //     GoldilocksField(16383556155787439553),
        //     GoldilocksField(18045582516498195573),
        //     GoldilocksField(7296969412159674050),
        //     GoldilocksField(8317318176954617326)
        // ];
        // // let res = PoseidonHash::hash_no_pad(&inputs);
        // // let res = hash_n_to_m_no_pad::<F, PoseidonPermutation>(&inputs, 4);
        //
        // let mut state = [F::ZERO; 12];
        //
        // // Absorb all input chunks.
        // state[..inputs.len()].copy_from_slice(&inputs);
        // state = F::poseidon(state);
        //
        // let res = state.into_iter().take(4).collect::<Vec<_>>();
        //
        // // let res = HashOut::from_vec(res);
        // let hex_string: String = unsafe{*std::mem::transmute::<*const _, *const [u8;32]>(res.as_ptr())}.iter().map(|byte| format!("{:02x}", byte)).collect();
        // let result: String = hex_string.chars()
        //     .collect::<Vec<char>>()
        //     .chunks(16)
        //     .map(|chunk| chunk.iter().collect::<String>())
        //     .collect::<Vec<String>>()
        //     .join(", ");
        // println!("cpu hash: {}", result);
        //
        // exit(0);

        let eddsa_proof = prove_ed25519::<F, C, D>(
            decode_hex(&args.msg.unwrap())?.as_slice(),
            decode_hex(&args.sig.unwrap())?.as_slice(),
            decode_hex(&args.pk.unwrap())?.as_slice(),
        )?;
        println!("Num public inputs: {}", eddsa_proof.2.num_public_inputs);

        return Ok(());

        // TODO: remove this double recursion will cause leaf proving error, why?
        let standard_config = CircuitConfig::standard_recursion_config();
        let (inner_proof, inner_vd, inner_cd) =
            recursive_proof::<F, C, C, D>(&eddsa_proof, None, &standard_config, None)?;
        println!("Num public inputs: {}", inner_cd.num_public_inputs);

        // recursively prove in a leaf
        let mut common_data = common_data_for_recursion::<GoldilocksField, C, D>();
        let mut builder = CircuitBuilder::<F, D>::new(standard_config.clone());
        let leaf_targets = builder.tree_recursion_leaf::<C>(inner_cd, &mut common_data)?;
        let data = builder.build::<C>();
        let leaf_vd = &data.verifier_only;
        let mut pw = PartialWitness::new();
        let leaf_data = TreeRecursionLeafData {
            inner_proof: &inner_proof,
            inner_verifier_data: &inner_vd,
            verifier_data: leaf_vd,
        };
        set_tree_recursion_leaf_data_target(&mut pw, &leaf_targets, &leaf_data)?;
        let leaf_proof = data.prove(pw)?;
        check_tree_proof_verifier_data(&leaf_proof, leaf_vd, &common_data)
            .expect("Leaf public inputs do not match its verifier data");
        data.verify(leaf_proof.clone()).expect("verify error");
        println!("Num public inputs: {}", common_data.num_public_inputs);

        let proof_bytes = leaf_proof.to_bytes();
        info!("Export proof: {} bytes", proof_bytes.len());

        println!(
            "Exporting root proof: {}",
            args.output_path
                .clone()
                .into_os_string()
                .into_string()
                .unwrap()
        );
        let mut file = File::create(args.output_path)?;
        file.write_all(&*proof_bytes)
            .expect("Leaf proof file write err");

        Ok(())
    }
}
