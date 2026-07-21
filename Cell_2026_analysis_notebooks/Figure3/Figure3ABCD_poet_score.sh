# Run this in the PoET root directory https://github.com/OpenProteinAI/PoET
python scripts/score.py \
    --msa_a3m_path ~/MSA_Pairformer/Cell_2026_analysis_notebooks/Figure3/data/Figure3_toxin_antitoxin/ParED_hhfilter.a3m \
    --variants_fasta_path ~/MSA_Pairformer/Cell_2026_analysis_notebooks/Figure3/data/Figure3_toxin_antitoxin/variant_sequences.pare_pard.fasta \
    --output_npy_path ~/MSA_Pairformer/Cell_2026_analysis_notebooks/Figure3/results/poet_toxin_antitoxin_scores.npy