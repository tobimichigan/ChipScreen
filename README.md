# ChipScreen
A Hybrid Fine-Tuned LLM + Agentic AI

# Project title and summary

<p>ChipScreen is an end-to-end software system that takes raw organ-on-a-chip (OoC) and cell-painting images plus plate metadata as input, and produces QC-checked, explained toxicity and phenotype reports as output with an agent layer that plans the analysis, grounds findings in mechanistic literature, and proposes the next wet-lab experiment. </p>

<p>The system is built from four trained model components ChipTox (multi-task toxicity classification ensemble), ChipCell (cell segmentation + phenotype clustering), GlowFromLight (bright-field → fluorescence image translation), and a QLoRA-fine-tuned Qwen3-4B language model orchestrated by a lightweight planning agent and surfaced through a web interface (desktop and mobile). It directly addresses three of the challenge's named task themes (drug toxicity prediction and response evaluation; bright-field-to-fluorescence image translation; single-cell phenotype analysis) in a single, interoperable pipeline rather than as three disconnected models.</p>


<img width="2315" height="593" alt="glow_metrics_by_split" src="https://github.com/user-attachments/assets/41c64a48-4408-4f9a-94fe-7fab140b604c" />
<img width="2112" height="3595" alt="glow_holdout_examples" src="https://github.com/user-attachments/assets/9fc623fc-92fa-41bc-a1ea-e1c204dcd0a3" />
<img width="2343" height="684" alt="eda_chiptox_chem_overview" src="https://github.com/user-attachments/assets/f3147074-87d0-4e0c-b760-a280a2bb8e96" />
<img width="2326" height="716" alt="eda_chiptox_chem_embedding" src="https://github.com/user-attachments/assets/933a91fc-89fc-4bde-88a2-9784528bfe5b" />
<img width="2059" height="622" alt="dosecurve_plate_qc_summary" src="https://github.com/user-attachments/assets/6650a058-b182-4d10-94a2-5915e4ad9302" />
<img width="2741" height="566" alt="dosecurve_plate_heatmaps" src="https://github.com/user-attachments/assets/742fe2f9-7745-49e4-8d51-0246ea7c3cf7" />
<img width="807" height="684" alt="chiptox_uncertainty_vs_error" src="https://github.com/user-attachments/assets/375809c1-4bf3-40a0-856a-1307df2d9d91" />
<img width="1923" height="989" alt="chiptox_per_task_auroc" src="https://github.com/user-attachments/assets/b293b7f9-77f0-4da5-9341-566949e78ae4" />
<img width="2439" height="745" alt="chiptox_holdout_roc_pr_calibration" src="https://github.com/user-attachments/assets/6e651807-a7c4-4ac0-bf7b-2d298b800741" />
<img width="607" height="593" alt="chiptox_holdout_confusion" src="https://github.com/user-attachments/assets/1f63c84f-a153-45f1-912e-e82050fdc8e8" />
<img width="2191" height="655" alt="chipcell_seg_metrics_by_split" src="https://github.com/user-attachments/assets/e7a5db4a-cc37-4ba0-bb26-1b5245f80803" />
<img width="2398" height="1168" alt="chipcell_holdout_segmentations" src="https://github.com/user-attachments/assets/086cac9b-8ae0-4bf2-b483-7f0a7c932c25" />
<img width="1815" height="807" alt="chipcell_cell_atlas" src="https://github.com/user-attachments/assets/870c5a9a-a818-47dd-aa68-d362c61a7691" />
<img width="2209" height="622" alt="history_glow" src="https://github.com/user-attachments/assets/670db95a-52c5-4e76-b279-1cfc927e097f" />
<img width="2190" height="622" alt="history_chiptox_cnn_attn_fold1" src="https://github.com/user-attachments/assets/2ebbe017-9ebc-4d9a-83c8-82587b51479f" />
<img width="2190" height="622" alt="history_chipcell_seg" src="https://github.com/user-attachments/assets/acf769bc-78e7-45f7-a367-537c5331289f" />
<img width="2277" height="684" alt="glow_trust_and_generalisation" src="https://github.com/user-attachments/assets/e609eafd-1297-4961-834e-bb4308ef3521" />


# Full WriteUp: 
https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/writeups/chipscreen-a-hybrid-fine-tuned-llm-agentic-ai-s

# Citation

# DOI (Digital Object Identifier)https://doi.org/10.34740/kaggle/w/118077

Handsonlabs Software Academy. (2026). ChipScreen: A Hybrid Fine-Tuned LLM + Agentic AI System for Organ-on-a-Chip. Kaggle. https://doi.org/10.34740/KAGGLE/W/118077
