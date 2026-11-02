"""
Stage 1 of 2-stage vector DB pipeline.

Reads all JSONL source files, splits text into overlapping chunks,
and writes the chunks evenly into N shard files — no GPU needed.

Output: /fsx/i777599/cache/vecdb_shards/shard_{i:04d}.jsonl
Each line: {"source": "/path/to/file.jsonl", "line": 42, "text": "...chunk..."}

Usage:
    python create_chunks.py --num_shards 30
    python create_chunks.py --num_shards 5 --max_docs 5000   # test
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_PATHS = [
    "/fsx/NOVA/training_data/cpt/v3/v3-training/odata_service_documentation.jsonl",
    "/fsx/NOVA/training_data/cpt/v3.5/v3.5-training/no_metadata/all_info_entity_sets_tool_12-08-2025_processed.jsonl",
    "/fsx/NOVA/training_data/cpt/v5/BKG/all_info_entity_sets_tools_sfsf_03-27-26_enriched.jsonl",
    "/fsx/NOVA/training_data/cpt/v5/SAP_learning/SAP_learning.jsonl",
    "/fsx/NOVA/rawri/reconstructed_data_with_description_parsed_text_only/consolidated/consolidated.jsonl",
    "/fsx/NOVA/training_data/cpt/v5/sap_help_formatted_updated_id.jsonl",
    "/fsx/NOVA/training_data/cpt/v5/help-portal/SAP_Help_BTP.jsonl",
    "/fsx/NOVA/training_data/cpt/v5/BTP_data_text_only/btp_pdfs.jsonl",
    "/fsx/NOVA/training_data/cpt/v5/BTP_GH_no_metadata/product-standards.jsonl",
    "/fsx/NOVA/training_data/cpt/v5/BTP_GH_no_metadata/rcs-docs.jsonl",
    "/fsx/NOVA/training_data/cpt/v5/ecc_tables_business_process_data_no_metadata_text_only/ecc_objects.jsonl",
    "/fsx/NOVA/training_data/cpt/v5/ecc_tables_business_process_data_no_metadata_text_only/ecc_tables.jsonl",
    "/fsx/NOVA/training_data/cpt/v5/ecc_tables_business_process_data_no_metadata_text_only/business_process.jsonl",
    "/fsx/NOVA/training_data/cpt/v5/ekg_training_data/sapedia/sappedia_verbalized.jsonl",
    "/fsx/NOVA/training_data/cpt/v5/ekg_training_data/service_and_support/verbalized_service_entity_instances.jsonl",
    "/fsx/NOVA/training_data/cpt/v5/ekg_training_data/service_and_support/verbalized_support_components.jsonl",
    "/fsx/NOVA/training_data/cpt/v5/ekg_training_data/ACH/ACH.jsonl",
    "/fsx/NOVA/training_data/cpt/v3.5/v3.5_deduped_merged/no_metadata/sap_activate_updated.jsonl",
    "/fsx/NOVA/training_data/cpt/v3.5/v3.5_deduped_merged/no_metadata/sap_cap_docs.jsonl",
    "/fsx/NOVA/training_data/cpt/v3.5/v3.5_deduped_merged/no_metadata/sap_fiori_apps.jsonl",
    "/fsx/NOVA/training_data/cpt/v3.5/v3.5_deduped_merged/no_metadata/sap_fiori_apps_batched.jsonl",
    "/fsx/NOVA/training_data/cpt/v3.5/v3.5_deduped_merged/no_metadata/sap_j4d_abap_ccm.jsonl",
    "/fsx/NOVA/training_data/cpt/v3.5/v3.5_deduped_merged/no_metadata/sap_j4d_api_assistant_all.jsonl",
    "/fsx/NOVA/training_data/cpt/v3.5/v3.5_deduped_merged/no_metadata/sap_leanix.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/american_productivity_and_quality_center_pe_descriptions.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/american_productivity_and_quality_center_pm_descriptions.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/application_component_ach_.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/isp_bc_descriptions.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/isp_mbc_descriptions.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/isp_sc_descriptions.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/leanix_leanix_descriptions.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/sap_fiori_apps_reference_library_descriptions.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/sap_isbn_descriptions_sisbn_uact.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/sap_key_performance_indicators_descriptions_skpi_kpi.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/sap_mxp_core_model_cr_descriptions.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/sap_mxp_core_model_pv_descriptions.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/sap_mxp_jarvis_descriptions.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/sap_product_and_production_management_system_descriptions_sisbn_uact.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/sap_product_and_production_management_system_product_descriptions.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/sap_s4hana_global_technical_name_catalog_cdsv_descriptions.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/sap_signavio_professional_services_sspia-mtr_descriptions.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/sap_signavio_strategic_objectives_descriptions.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/sap_solution_area_hierarchy_descriptions_csl.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/sap_solution_area_hierarchy_descriptions_lgp.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/sapedia_sappedia_.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/service_and_support_catalogue_service_component_descriptions.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/service_and_support_catalogue_service_descriptions.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/ssmb_descriptions.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/ssmb_descriptions_b.jsonl",
    "/fsx/EKG_training_data/EKG_schema_converted/combined_jsonl/value_chain_group_vgp_descriptions.jsonl",
]


def char_chunks(text, chunk_chars, overlap_chars):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_chars])
        start += chunk_chars - overlap_chars
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_shards",    type=int, default=30)
    parser.add_argument("--chunk_size",    type=int, default=512,
                        help="Chunk size in tokens (1 token ≈ 4 chars)")
    parser.add_argument("--chunk_overlap", type=int, default=64)
    parser.add_argument("--output_dir",    type=str,
                        default="/fsx/i777599/cache/vecdb_shards")
    parser.add_argument("--max_docs",      type=int, default=None,
                        help="Cap total source docs (for testing)")
    args = parser.parse_args()

    chunk_chars   = args.chunk_size   * 4
    overlap_chars = args.chunk_overlap * 4

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write a config file so embedding jobs know the shard count
    config = {
        "num_shards": args.num_shards,
        "chunk_size_tokens": args.chunk_size,
        "chunk_overlap_tokens": args.chunk_overlap,
    }
    with open(out_dir / "shards_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Open all shard file handles upfront
    shard_handles = [
        open(out_dir / f"shard_{i:04d}.jsonl", "w") for i in range(args.num_shards)
    ]

    total_chunks = 0
    total_docs   = 0

    try:
        for path in DATA_PATHS:
            p = Path(path)
            if not p.exists():
                log.warning("Skipping (not found): %s", path)
                continue
            log.info("Reading: %s", p.name)

            with open(p) as fh:
                for line_idx, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = obj.get("text", "")
                    if not text:
                        continue

                    for chunk in char_chunks(text, chunk_chars, overlap_chars):
                        shard_idx = total_chunks % args.num_shards
                        entry = {"source": path, "line": line_idx, "text": chunk}
                        shard_handles[shard_idx].write(
                            json.dumps(entry, ensure_ascii=False) + "\n"
                        )
                        total_chunks += 1

                    total_docs += 1
                    if args.max_docs and total_docs >= args.max_docs:
                        log.info("Reached --max_docs=%d, stopping.", args.max_docs)
                        break

            if args.max_docs and total_docs >= args.max_docs:
                break

    finally:
        for fh in shard_handles:
            fh.close()

    # Update config with actual totals
    config["total_chunks"] = total_chunks
    config["total_docs"]   = total_docs
    with open(out_dir / "shards_config.json", "w") as f:
        json.dump(config, f, indent=2)

    log.info("Done. Total docs: %d, total chunks: %d", total_docs, total_chunks)
    log.info("Shards written to: %s", out_dir)

    # Print shard sizes
    for i in range(args.num_shards):
        shard_path = out_dir / f"shard_{i:04d}.jsonl"
        size_mb = shard_path.stat().st_size / 1e6
        log.info("  shard_%04d.jsonl : %.1f MB", i, size_mb)


if __name__ == "__main__":
    main()
