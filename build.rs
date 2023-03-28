use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Result;

fn get_dxc_path() -> Result<PathBuf> {
    if cfg!(linux) {
        return Ok(PathBuf::from("/usr/bin/dxc"));
    } else {
        return Ok(env::var("VULKAN_SDK").map(|sdk| PathBuf::from(&sdk).join("Bin/dxc"))?);
    }
}

enum Stage {
    Vertex,
    Fragment,
}

#[allow(unreachable_patterns)]
fn hlsl_profile(stage: Stage) -> Result<String> {
    Ok(match stage {
        Stage::Vertex => "vs",
        Stage::Fragment => "ps",
        _ => todo!(),
    }
        .to_owned()
        + "_6_7")
}

fn compile_hlsl(path: &Path, out: &Path, stage: Stage) -> Result<()> {
    let dxc = get_dxc_path()?;
    let output = Command::new(dxc)
        // Entry point: 'main'
        .arg("-E main")
        // Output file
        .arg("-Fo".to_owned() + out.to_str().unwrap())
        // HLSL version 2021
        .arg("-HV 2021")
        // HLSL profile depending on shader stage
        .arg("-T ".to_owned() + &hlsl_profile(stage)?)
        // Emit SPIR-V reflection info.
        // Note that we disable this for now, because this causes DXC to emit the SPV_GOOGLE_hlsl_functionality1 extension,
        // which we then have to enable in Vulkan. This is possible, but not really desired and ash does not support it, so preferably
        // reflection just works without this flag too.
        // .arg("-fspv-reflect")
        // SPIR-V target env
        .arg("-fspv-target-env=vulkan1.3")
        // Actually generate SPIR-V
        .arg("-spirv")
        // Our input file
        .arg(path)
        .output()?;
    println!(
        "Error compiling shader: {}",
        String::from_utf8(output.stderr).unwrap()
    );
    Ok(())
}

fn main() {
    println!("cargo:rerun-if-changed=src/shaders/src/vert.hlsl");
    println!("cargo:rerun-if-changed=src/shaders/src/frag.hlsl");

    compile_hlsl(
        "src/shaders/src/vert.hlsl".as_ref(),
        "src/shaders/spv/vert.spv".as_ref(),
        Stage::Vertex,
    )
        .unwrap();
    compile_hlsl(
        "src/shaders/src/frag.hlsl".as_ref(),
        "src/shaders/spv/frag.spv".as_ref(),
        Stage::Fragment,
    )
        .unwrap();
}
