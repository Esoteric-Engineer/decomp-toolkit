//! Bakes `r13`-relative small-data accesses into compiled objects.
//!
//! Some games (e.g. those linked with SN Systems' `ngcld`) address *all* small data through `r13`, including `.sdata2`/`.sbss2`.
//! `mwld` always resolves `R_PPC_EMB_SDA21` relocations against those sections via `r2`, so compiled code referencing them fails to link/match.
//! `dtk dol split` records the addresses needed to resolve these relocations ahead of the link.
//! `dtk elf bake-sda` rewrites each affected instruction to use `r13` with a fixed displacement, removing the relocation.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow, bail, ensure};
use object::elf;
use serde::{Deserialize, Serialize};

use crate::obj::ObjInfo;

/// Sections `mwld` would resolve via `r2`.
const SDA2_SECTIONS: [&str; 2] = [".sdata2", ".sbss2"];

fn is_sda2_section(name: &str) -> bool {
    SDA2_SECTIONS.contains(&name.split(':').next().unwrap_or(name))
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdaBakeData {
    /// Value of `_SDA_BASE_` (`r13`).
    pub sda_base: u32,
    /// Address of every global symbol located in `.sdata2`/`.sbss2`, by name.
    pub symbols: BTreeMap<String, u32>,
    /// Start address of each unit's `.sdata2`/`.sbss2` split, by unit name and section name.
    pub units: BTreeMap<String, BTreeMap<String, u32>>,
}

impl SdaBakeData {
    pub fn from_obj(obj: &ObjInfo) -> Result<Self> {
        let sda_base = obj.sda_base.ok_or_else(|| anyhow!("_SDA_BASE_ is unknown"))?;
        let mut data = SdaBakeData { sda_base, ..Default::default() };
        for (section_index, section) in obj.sections.iter() {
            if !is_sda2_section(&section.name) {
                continue;
            }
            let section_name = section.name.split(':').next().unwrap_or(&section.name);
            for (_, symbol) in obj.symbols.for_section(section_index) {
                // Only globals can be referenced from other objects
                if symbol.name.is_empty() || symbol.flags.is_local() {
                    continue;
                }
                data.symbols.insert(symbol.name.clone(), symbol.address as u32);
            }
            for (addr, split) in section.splits.iter() {
                data.units
                    .entry(split.unit.clone())
                    .or_default()
                    .entry(section_name.to_string())
                    .or_insert(addr);
            }
        }
        Ok(data)
    }
}

struct SectionHeader {
    name: String,
    kind: u32,
    offset: usize,
    size: usize,
    link: u32,
    info: u32,
    entsize: usize,
}

fn read_u16(data: &[u8], offset: usize) -> Result<u16> {
    let bytes = data.get(offset..offset + 2).ok_or_else(|| anyhow!("Truncated ELF"))?;
    Ok(u16::from_be_bytes(bytes.try_into().unwrap()))
}

fn read_u32(data: &[u8], offset: usize) -> Result<u32> {
    let bytes = data.get(offset..offset + 4).ok_or_else(|| anyhow!("Truncated ELF"))?;
    Ok(u32::from_be_bytes(bytes.try_into().unwrap()))
}

fn write_u32(data: &mut [u8], offset: usize, value: u32) {
    data[offset..offset + 4].copy_from_slice(&value.to_be_bytes());
}

fn read_str(data: &[u8], offset: usize) -> Result<String> {
    let bytes = data.get(offset..).ok_or_else(|| anyhow!("Truncated ELF"))?;
    let end = bytes.iter().position(|&b| b == 0).ok_or_else(|| anyhow!("Unterminated string"))?;
    Ok(String::from_utf8_lossy(&bytes[..end]).into_owned())
}

fn read_section_headers(data: &[u8]) -> Result<(usize, usize, Vec<SectionHeader>)> {
    ensure!(data.get(..4) == Some(&elf::ELFMAG[..]), "Not an ELF file");
    ensure!(
        data.get(4) == Some(&elf::ELFCLASS32) && data.get(5) == Some(&elf::ELFDATA2MSB),
        "Expected a 32-bit big-endian ELF"
    );
    ensure!(read_u16(data, 16)? == elf::ET_REL, "Expected a relocatable object");
    let shoff = read_u32(data, 32)? as usize;
    let shentsize = read_u16(data, 46)? as usize;
    let shnum = read_u16(data, 48)? as usize;
    let shstrndx = read_u16(data, 50)? as usize;
    ensure!(shentsize >= 40, "Invalid section header size");
    let mut headers = Vec::with_capacity(shnum);
    let mut name_offsets = Vec::with_capacity(shnum);
    for i in 0..shnum {
        let base = shoff + i * shentsize;
        name_offsets.push(read_u32(data, base)? as usize);
        headers.push(SectionHeader {
            name: String::new(),
            kind: read_u32(data, base + 4)?,
            offset: read_u32(data, base + 16)? as usize,
            size: read_u32(data, base + 20)? as usize,
            link: read_u32(data, base + 24)?,
            info: read_u32(data, base + 28)?,
            entsize: read_u32(data, base + 36)? as usize,
        });
    }
    let strtab_offset =
        headers.get(shstrndx).ok_or_else(|| anyhow!("Invalid section name table"))?.offset;
    for (header, name_offset) in headers.iter_mut().zip(name_offsets) {
        header.name = read_str(data, strtab_offset + name_offset)?;
    }
    Ok((shoff, shentsize, headers))
}

/// Rewrites every `R_PPC_EMB_SDA21` relocation targeting `.sdata2`/`.sbss2` in the relocatable
/// object `data` as a fixed `r13`-relative access, and removes the relocation.
/// Returns the number of relocations baked.
pub fn bake_object(data: &mut [u8], bake: &SdaBakeData, unit: &str) -> Result<usize> {
    let (shoff, shentsize, headers) = read_section_headers(data)?;
    let mut baked = 0;
    for (rela_index, rela) in headers.iter().enumerate() {
        if rela.kind != elf::SHT_RELA || rela.size == 0 {
            continue;
        }
        let entsize = if rela.entsize == 0 { 12 } else { rela.entsize };
        ensure!(entsize >= 12 && rela.size % entsize == 0, "Invalid relocation section");
        let target = headers
            .get(rela.info as usize)
            .ok_or_else(|| anyhow!("Invalid relocation target section"))?;
        let symtab =
            headers.get(rela.link as usize).ok_or_else(|| anyhow!("Invalid symbol table"))?;
        let strtab = headers
            .get(symtab.link as usize)
            .ok_or_else(|| anyhow!("Invalid string table"))?
            .offset;
        let sym_entsize = if symtab.entsize == 0 { 16 } else { symtab.entsize };

        let mut kept = Vec::with_capacity(rela.size / entsize);
        let mut patches = Vec::new();
        for entry_offset in (rela.offset..rela.offset + rela.size).step_by(entsize) {
            let entry = data[entry_offset..entry_offset + entsize].to_vec();
            let r_offset = read_u32(data, entry_offset)? as usize;
            let r_info = read_u32(data, entry_offset + 4)?;
            let r_addend = read_u32(data, entry_offset + 8)? as i32;
            if r_info & 0xFF != elf::R_PPC_EMB_SDA21 {
                kept.push(entry);
                continue;
            }
            let sym = symtab.offset + (r_info >> 8) as usize * sym_entsize;
            let st_name = read_u32(data, sym)? as usize;
            let st_value = read_u32(data, sym + 4)?;
            let st_shndx = read_u16(data, sym + 14)?;
            let name = read_str(data, strtab + st_name)?;
            let address = if st_shndx == elf::SHN_UNDEF {
                match bake.symbols.get(&name) {
                    Some(&address) => address,
                    None => {
                        kept.push(entry);
                        continue;
                    }
                }
            } else {
                let Some(section) = headers.get(st_shndx as usize) else {
                    kept.push(entry);
                    continue;
                };
                if !is_sda2_section(&section.name) {
                    kept.push(entry);
                    continue;
                }
                let start = bake
                    .units
                    .get(unit)
                    .and_then(|sections| sections.get(&section.name))
                    .ok_or_else(|| {
                    anyhow!(
                        "Symbol '{}' is in {}, but unit '{}' has no {} split",
                        name,
                        section.name,
                        unit,
                        section.name
                    )
                })?;
                start.wrapping_add(st_value)
            };
            let disp = address.wrapping_add(r_addend as u32).wrapping_sub(bake.sda_base) as i32;
            let Ok(disp) = i16::try_from(disp) else {
                bail!(
                    "Symbol '{}' at {:#010X} is out of range of _SDA_BASE_ ({:#010X})",
                    name,
                    address,
                    bake.sda_base
                );
            };
            // mwcc points the relocation at the low halfword of the instruction
            patches.push((target.offset + (r_offset & !3), disp));
        }
        if patches.is_empty() {
            continue;
        }
        for (ins_offset, disp) in patches {
            let ins = read_u32(data, ins_offset)?;
            write_u32(data, ins_offset, (ins & !0x1F_FFFF) | (13 << 16) | (disp as u16 as u32));
            baked += 1;
        }
        // Compact the remaining relocations and shrink the section. Nothing else moves.
        let mut cursor = rela.offset;
        for entry in &kept {
            data[cursor..cursor + entsize].copy_from_slice(entry);
            cursor += entsize;
        }
        data[cursor..rela.offset + rela.size].fill(0);
        write_u32(data, shoff + rela_index * shentsize + 20, (kept.len() * entsize) as u32);
    }
    Ok(baked)
}

#[cfg(test)]
mod test {
    use object::{
        Architecture, BinaryFormat, Endianness, RelocationFlags, SectionKind, SymbolFlags,
        SymbolKind, SymbolScope,
        write::{Object, Relocation, Symbol, SymbolSection},
    };

    use super::*;

    #[test]
    fn bake_rewrites_only_sda2_relocations() {
        let mut obj = Object::new(BinaryFormat::Elf, Architecture::PowerPc, Endianness::Big);
        let text = obj.add_section(vec![], b".text".to_vec(), SectionKind::Text);
        // lwz r3, 0(r0) ×3, lfs f1, 0(r0)
        obj.append_section_data(
            text,
            &[
                0x80, 0x60, 0x00, 0x00, 0x80, 0x60, 0x00, 0x00, 0x80, 0x60, 0x00, 0x00, 0xC0, 0x20,
                0x00, 0x00,
            ],
            4,
        );
        let sdata2 = obj.add_section(vec![], b".sdata2".to_vec(), SectionKind::ReadOnlyData);
        obj.append_section_data(sdata2, &[0; 8], 4);
        let extern_sym = |name: &str| Symbol {
            name: name.as_bytes().to_vec(),
            value: 0,
            size: 0,
            kind: SymbolKind::Data,
            scope: SymbolScope::Dynamic,
            weak: false,
            section: SymbolSection::Undefined,
            flags: SymbolFlags::None,
        };
        let gx_data = obj.add_symbol(extern_sym("__GXData"));
        let sdata_var = obj.add_symbol(extern_sym("someSdataVar"));
        let float_const = obj.add_symbol(Symbol {
            name: b"@123".to_vec(),
            value: 4,
            size: 4,
            kind: SymbolKind::Data,
            scope: SymbolScope::Compilation,
            weak: false,
            section: SymbolSection::Section(sdata2),
            flags: SymbolFlags::None,
        });
        let sda21 = |offset, symbol, addend| Relocation {
            offset,
            symbol,
            addend,
            flags: RelocationFlags::Elf { r_type: elf::R_PPC_EMB_SDA21 },
        };
        // mwcc-style offset (instruction + 2)
        obj.add_relocation(text, sda21(2, gx_data, 0)).unwrap();
        obj.add_relocation(text, sda21(4, sdata_var, 0)).unwrap();
        obj.add_relocation(text, sda21(8, gx_data, 4)).unwrap();
        obj.add_relocation(text, sda21(14, float_const, 0)).unwrap();
        let mut data = obj.write().unwrap();

        let bake = SdaBakeData {
            sda_base: 0x806734E0,
            symbols: BTreeMap::from([("__GXData".to_string(), 0x80670788)]),
            units: BTreeMap::from([(
                "unit.c".to_string(),
                BTreeMap::from([(".sdata2".to_string(), 0x80670000)]),
            )]),
        };
        assert_eq!(bake_object(&mut data, &bake, "unit.c").unwrap(), 3);

        let file = object::File::parse(&*data).unwrap();
        use object::{Object as _, ObjectSection as _};
        let text = file.section_by_name(".text").unwrap();
        let bytes = text.data().unwrap();
        let ins = |i: usize| u32::from_be_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        // 0x80670788 - 0x806734E0 = -0x2D58
        assert_eq!(ins(0), 0x806DD2A8);
        assert_eq!(ins(1), 0x80600000);
        assert_eq!(ins(2), 0x806DD2AC);
        // 0x80670004 - 0x806734E0 = -0x34DC
        assert_eq!(ins(3), 0xC02DCB24);
        let relocs: Vec<_> = text.relocations().map(|(offset, _)| offset).collect();
        assert_eq!(relocs, vec![4]);

        // Missing unit split is an error
        let mut data = obj.write().unwrap();
        assert!(bake_object(&mut data, &bake, "other.c").is_err());
    }
}
