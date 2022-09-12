Step-by-step implementations:

**A) EEG part - Facial representations and repetition suppression in human brains**:

Step 1: EEG classification-based decoding
> run: eeg/decoding.py  
> output: eeg/decoding_acc  
> plot: plot/eeg/decoding  
> figure: plot/eeg/decoding  

Step 2: decoding-based cross-temporal RDMs
> run: eeg/acc-ctrdmsd_cal.py  
> output: eeg/acc-ctrdms

Step 3: Make model RDMs
> run: models/eeg/  
> output: models/eeg/modelrdm
> plot: plot/model/eeg/plot_model_rdms.py & plot/model/dcnn/plot_model_rdms.py
> figure: plot/model/eeg/ & plot/model/dcnn/

Step 4: cross-temporal RSA
> run: eeg-model/acc-ctcorrs_cal.py  
> output: eeg-model/acc-ctcorrs
> plot: plot/eeg-model/plot_tbt_corrs.py & plot_ct_corrs.py
> figure: plot/eeg-model/tbt_results & ct_results

**B) DCNN part1 - Facial representations in DCNN**:

Step 1: Get activations from VGG-face & non-trained VGG
> run: dcnn/getfeatures.py & dcnn/getfeatures_random.py  
> output: dcnn/features & dcnn/feature_random

Step 2: Classification based on activations of VGG-face & non-trained VGG
> run: dcnn/decoding.py & dcnn/decoding_random.py  
> output: dcnn/classification_results/acc.txt & dcnn/classification_results/acc_random.txt
> plot: plot/dcnn/plot_decoding_acc_3faces.py
> figure: plot/dcnn/decoding/accs.jpg

Step 3: Calculate VGG-face RDMs & non-trained VGG RDMs (450 * 450 RDM for each layer) after PCA
> run: dcnn/getrdms_afterpca.py & dcnn/getrdms_random_afterpca.py  
> output: dcnn/rdms_afterPCA & dcnn/rdms_random_afterPCA
> plot: plot/dcnn/plot_rdm.py
> figure: plot/dcnn/rdms_afterPCA

Step 4: Make model RDMs
> run: models/dcnn/  
> output: models/dcnn/modelrdm
> plot: plot/model/dcnn/plot_model_rdms.py  
> figure: plot/model/dcnn

Step 5: RSA between DCNN RDMs and model RDMs
> run: dcnn-model/corrs_cal.py  
> output: dcnn-model/corrs
> plot: plot/dcnn-model/plot_corrs.py  
> figure: plot/dcnn-model/results

**C) DCNN part2 - Modified DCNNs with two RS mechanisms**:

Step 6: Calculate Fatigue-modified DCNNs RDMs
> run: dcnn/getrdms_fatigue.py, getrdms_fatigue_3faces.py, getrdms_random_fatigue.py, getrdms_random_fatigue_3faces.py  
> output: dcnn/rdms_fatigue, rdms_fatigue_3faces, rdms_random_fatigue, rdms_random_fatigue_3faces  
> plot: plot/dcnn/plot_rdm_1350.py  
> figure: plot/dcnn/modifiedrdms

Step 7: Calculate Sharpening-modified DCNNs
> run: dcnn/getrdms_sharpening.py, getrdms_sharpening_3faces.py, getrdms_random_sharpening.py, getrdms_random_sharpening_3faces.py  
> output: dcnn/rdms_sharpening, rdms_sharpening_3faces, rdms_random_sharpening, rdms_random_sharpening_3faces  
> plot: plot/dcnn/plot_rdm_1350.py  
> figure: plot/dcnn/modifiedrdms

**D) EEG-DCNN part - Comparisons between brains and DCNNs**:

Step 8: time-by-time RSA results between EEG and DCNNS for 8 layers
> run: eeg-dcnn/corrs_cal_*.py  
> output: eeg-dcnn/corrs  

Step 9: cross-temporal RSA results between EEG and DCNN for layer 16
> run: eeg-dcnn/ctcorrs_cal_*_ly16.py  
> output: eeg-dcnn/ctcorrs  
