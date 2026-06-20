# Literature Review and Design Mapping

The project uses primary papers and official bibliographic records. Citation
counts are not used as evidence of correctness; they were used only to help
prioritize influential work during discovery.

## Personalized Glucose Forecasting

1. Yang et al., "Personalized Blood Glucose Prediction for Type 1 Diabetes
   Using Evidential Deep Learning and Meta-Learning," IEEE TBME, 2023,
   DOI `10.1109/TBME.2022.3187703`. This is the closest precedent: an
   attention RNN, evidential output, and MAML enable fast adaptation with
   limited subject data. HAPF should compare directly with this adaptation
   strategy rather than assume LoRA is superior.
2. Daniels et al., "A Multitask Learning Approach to Personalized Blood
   Glucose Prediction," IEEE JBHI, 2022, DOI
   `10.1109/JBHI.2021.3100558`. It supports partial pooling across subjects and
   reports performance with limited subject-specific training data.
3. Li et al., "GluNet: A Deep Learning Framework for Accurate Glucose
   Forecasting," IEEE JBHI, 2020, DOI `10.1109/JBHI.2019.2931842`. Its causal
   dilated CNN motivates the compact first backbone and 30/60-minute targets.
4. Marling and Bunescu, "The OhioT1DM Dataset for Blood Glucose Level
   Prediction: Update 2020," CEUR Workshop Proceedings, PMID `33584164`.
   It documents eight weeks of CGM and event data for 12 people and defines a
   core benchmark represented in the sample archive.
5. Zhu et al., "Deep transfer learning and data augmentation improve glucose
   levels prediction in type 2 diabetes patients," npj Digital Medicine, 2021,
   DOI `10.1038/s41746-021-00480-x`. It motivates transfer-learning baselines
   and cautions against comparing personalization only to patient-only models.

## Adaptation Mechanisms

6. Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation of Deep
   Networks," ICML, 2017, arXiv `1703.03400`. MAML supplies the episodic
   support/query protocol and is a required baseline.
7. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR,
   2022, arXiv `2106.09685`. HAPF borrows low-rank parameterization, not the
   assumption that LLM adaptation transfers unchanged to physiological time
   series.
8. Ha et al., "HyperNetworks," ICLR, 2017, arXiv `1609.09106`. This motivates
   generating patient adapter coefficients from static and early-history
   context.
9. Shamsian et al., "Personalized Federated Learning using Hypernetworks,"
   ICML, 2021. The patient-conditioned shared hypernetwork is a close structural
   precedent for cold-start personalization.
10. Li et al., "Ditto: Fair and Robust Federated Learning Through
    Personalization," ICML, 2021, arXiv `2012.04221`. Ditto motivates explicit
    population fallback, regularized local objectives, and per-client harm
    reporting.
11. Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer,"
    AAAI, 2018, DOI `10.1609/aaai.v32i1.11671`. Feature-wise conditioning is a
    low-cost static-context baseline.

## Clinical Time Series and Uncertainty

12. Che et al., "Recurrent Neural Networks for Multivariate Time Series with
    Missing Values," Scientific Reports, 2018, DOI
    `10.1038/s41598-018-24271-9`. GRU-D establishes that missingness and time
    since observation can be informative and must be modeled explicitly.
13. Lim et al., "Temporal Fusion Transformers for interpretable multi-horizon
    time series forecasting," International Journal of Forecasting, 2021, DOI
    `10.1016/j.ijforecast.2021.03.012`. TFT is a larger-data comparator for
    static context, observed covariates, and multi-horizon quantiles.
14. Montero-Manso and Hyndman, "Principles and algorithms for forecasting
    groups of time series: Locality and globality," International Journal of
    Forecasting, 2021, DOI `10.1016/j.ijforecast.2021.03.004`. It frames the
    global-versus-local model tradeoff that HAPF addresses through partial
    pooling.
15. Gibbs and Candes, "Adaptive Conformal Inference Under Distribution Shift,"
    NeurIPS, 2021, arXiv `2106.00170`. It motivates online coverage monitoring
    under patient drift after the simpler split-conformal baseline.

## Personalized Normative Modeling

16. Marquand et al., "Understanding Heterogeneity in Clinical Cohorts Using
    Normative Models: Beyond Case-Control Studies," Biological Psychiatry,
    2016, DOI `10.1016/j.biopsych.2015.12.023`. It directly supports replacing
    a single case-control boundary with subject-level deviation modeling.
17. Kia et al., "Closing the life-cycle of normative modeling using federated
    hierarchical Bayesian regression," PLOS ONE, 2022, DOI
    `10.1371/journal.pone.0278776`. It motivates hierarchical partial pooling,
    site effects, and lifecycle calibration.

## Design Conclusion

The literature supports personalization, but it does not establish that one
LoRA per patient and variable is optimal. The defensible experiment is a
comparison among population, multitask, transfer, meta-learning, low-rank,
hypernetwork, and partial-pooling baselines under the same subject-safe split.

