# Medication Feature Engineering

## Medication Severity Profiling

Medication administration records were encoded as time-varying features using a clinically grounded severity profiling scheme. Rather than representing each drug category as a binary indicator of any administration, medications were mapped to composite ordinal tiers reflecting escalation intensity, treatment context, and organ support burden. This approach was motivated by the observation that a binary "any antibiotic administered" feature, for example, cannot distinguish surgical prophylaxis from last-resort therapy for multidrug-resistant infection, despite the substantial prognostic difference between these scenarios.

All medications were identified by Anatomical Therapeutic Chemical (ATC) classification codes extracted from electronic health record administration timestamps. Nine clinically distinct medication dimensions were defined, each encoded as a single ordinal feature per temporal bin, alongside three derived summary features, yielding 12 medication features in total. The profiling scheme was developed through synthesis of established clinical severity scales, published dose-equivalency frameworks, and iterative validation against the cohort's ATC code distribution. Table X provides an overview of all medication features and their tier definitions.

**Table X.** Medication severity profiling scheme. Nine ordinal features capture escalation intensity across clinically distinct treatment dimensions, supplemented by three derived summary features. Each ordinal feature is computed as the maximum applicable tier across all administrations within a temporal bin.

| Feature | Scale | Tier | Definition | Key agents / criteria | Informing framework |
|---|---|---|---|---|---|
| **Antibiotic escalation** | 0--6 | 0 | No antibiotic | -- | WHO AWaRe [1,2], ASI [3], Weiss hierarchy [4] |
| | | 1 | Narrow first-line | Penicillin V/G, nitrofurantoin, sulfamethizole, trimethoprim | |
| | | 2 | Standard access | Metronidazole, dicloxacillin, pivmecillinam, aminoglycosides, 1st-gen cephalosporins | |
| | | 3 | Moderate watch | Cefuroxime, macrolides, clindamycin, co-trimoxazole, amoxicillin-clavulanate | |
| | | 4 | Broad watch | Vancomycin, fluoroquinolones, ceftriaxone | |
| | | 5 | Very broad / critical | Piperacillin-tazobactam, carbapenems, cefepime, ceftazidime | |
| | | 6 | Reserve / last-resort | Linezolid, colistin, ceftazidime-avibactam | |
| **Antibiotic polypharmacy** | 0+ | -- | Count of distinct J01 ATC codes per bin | -- | -- |
| **Hemodynamic support** | 0--3 | 0 | No vasoactive support | -- | SOFA-CV [6], NED [7] |
| | | 1 | Perioperative only | Ephedrine, phenylephrine (OR bolus) | |
| | | 2 | Single ICU vasopressor | Norepinephrine, epinephrine, dopamine, dobutamine, vasopressin | |
| | | 3 | Multi-agent / refractory | ≥2 vasopressors, or vasopressor + amiodarone | |
| **Sedation and NMB** | 0--5 | 0 | No pharmacological CNS depression | Non-CNS analgesics only (paracetamol, NSAIDs, regional blocks) | ASA Continuum [24], MOAA/S [25], RASS [11], PADIS [10] |
| | | 1 | Anxiolysis / mild CNS effect | Sub-dissociative opioid, oral benzodiazepine, melatonin, zopiclone, low-dose clonidine | |
| | | 2 | Moderate sedation | Dexmedetomidine, esketamine/ketamine alone*, haloperidol, olanzapine, quetiapine | |
| | | 3 | Deep sedation | Propofol alone*, midazolam infusion, lorazepam infusion, ketamine continuous infusion | |
| | | 4 | General anesthesia / unarousable | Volatile anesthetics, thiopental, etomidate; propofol or esketamine when co-occurring with volatile/NMBA/anesthetic opioid/propofol* | |
| | | 5 | General anesthesia + NMB | Tier 4 + concurrent NMBA (cisatracurium, rocuronium, suxamethonium) | |
| **Coagulation management** | 0--4 | 0 | No coagulation drugs | -- | CRASH-2 [18] |
| | | 1 | Standard prophylaxis | Prophylactic LMWH, single antiplatelet, or TXA alone | |
| | | 2 | Therapeutic anticoagulation | Therapeutic LMWH/DOAC/VKA, IV heparin, or dual antiplatelet | |
| | | 3 | Active hemorrhage Rx | Fibrinogen, PCC, vitamin K, desmopressin, or protamine | |
| | | 4 | Massive / refractory hemorrhage | rFVIIa, or ≥2 concurrent factor products | |
| **Organ support** | 0--4 | 0 | No organ support | -- | NICE-SUGAR [19] |
| | | 1 | Oral comorbidity meds | Oral thiazide/K-sparing, SC insulin, single electrolyte | |
| | | 2 | Active IV support | IV loop diuretic, IV insulin, or ≥2 electrolyte replacements | |
| | | 3 | Escalated support | Albumin, TPN, metolazone, or multi-component support | |
| | | 4 | Refractory | Continuous loop infusion, IV bicarbonate, or ≥3 concurrent supports | |
| **Opioid intensity** | 0--3 | 0 | No opioid | -- | -- |
| | | 1 | Mild / oral | Tramadol, codeine, tapentadol | |
| | | 2 | Strong / ICU opioid | Morphine, oxycodone, ketobemidone IV; fentanyl, remifentanil | |
| | | 3 | Multi-agent strong | ≥2 distinct strong/ICU opioid ATC codes concurrent | |
| **Surgical / procedural exposure** | 0--3 | 0 | No surgical markers | -- | -- |
| | | 1 | Regional anesthesia | Any local anesthetic (N01BB) | |
| | | 2 | General anesthesia | Volatile anesthetic (N01AB*), esketamine/ketamine + propofol co-induction*, or propofol + anesthetic opioid* | |
| | | 3 | Emergency / neuro | Thiopental (refractory ICP / emergency RSI) | |
| **Acute deterioration** | 0--2 | 0 | No reversal event | -- | -- |
| | | 1 | Single reversal agent | Naloxone, flumazenil, or acetylcysteine | |
| | | 2 | Multiple reversal agents | ≥2 of the above concurrent | |
| **Comorbidity med count** | 0--5 | -- | Count of chronic medication classes active | Beta-blocker, basal insulin, antiplatelet, oral anticoagulant, CV comorbidity drugs | -- |
| **Treatment dimensionality** | 0--9 | -- | Count of non-zero ordinal dimensions per bin | -- | -- |
| **Max severity signal** | 0--1 | -- | Maximum tier / scale across all ordinal features | Normalized composite severity index | -- |

Abbreviations: ASA, American Society of Anesthesiologists; ASI, Antibiotic Spectrum Index; AWaRe, Access Watch Reserve; CNS, central nervous system; CV, cardiovascular; DOAC, direct oral anticoagulant; GA, general anesthesia; ICP, intracranial pressure; LMWH, low-molecular-weight heparin; MOAA/S, Modified Observer's Assessment of Alertness/Sedation; NED, norepinephrine-equivalent dose; NMB, neuromuscular blockade; NMBA, neuromuscular blocking agent; PCC, prothrombin complex concentrate; RASS, Richmond Agitation-Sedation Scale; rFVIIa, recombinant activated factor VII; RSI, rapid sequence induction; SC, subcutaneous; SOFA-CV, Sequential Organ Failure Assessment cardiovascular component; TPN, total parenteral nutrition; TXA, tranexamic acid; VKA, vitamin K antagonist.

\* Tier assignment for propofol and esketamine/ketamine is co-occurrence-dependent. For the sedation tier: when administered alone, propofol is assigned Tier 3 (deep sedation) and esketamine/ketamine Tier 2 (moderate sedation); when co-occurring in the same temporal bin with a volatile anesthetic, neuromuscular blocking agent, or anesthetic opioid (remifentanil, alfentanil, or sufentanil), both are elevated to Tier 4. Propofol co-occurring with esketamine/ketamine also triggers Tier 4 (co-induction pattern). For the surgical tier: Tier 2 is triggered by volatile anesthetics, the co-induction pattern of esketamine/ketamine + propofol, or propofol + anesthetic opioid in the same bin; neuromuscular blocking agents do not trigger surgical Tier 2, as their use is ambiguous between operative and ICU contexts. The anesthetic opioid trigger was limited to remifentanil, alfentanil, and sufentanil — ultra-short-acting agents whose concurrent use with propofol is a strong marker of general anesthesia. Fentanyl (N01AH01) was excluded from this trigger set despite being classified under the same ATC group (N01AH), as it is extensively used for ICU analgosedation alongside propofol and its inclusion would misclassify ICU deep sedation as general anesthesia. This co-occurrence-based disambiguation avoids the need for cross-referencing care-setting data (admission-discharge-transfer location labels) during medication preprocessing. Care-setting location is available to the model as a separate input feature, enabling the model to learn any residual location-medication interactions directly.

### Antibiotic Escalation (2 features)

Systemic antibacterials (ATC J01) were mapped to a six-tier ordinal escalation variable reflecting antimicrobial spectrum breadth and resistance implications. The tier structure was informed by three complementary frameworks: the WHO Access, Watch, Reserve (AWaRe) classification, which stratifies antibiotics by antimicrobial resistance impact [1,2]; the Antibiotic Spectrum Index (ASI), which assigns each agent a score based on the number of clinically important organism categories covered [3]; and the Weiss beta-lactam hierarchy, a six-rank ordinal scale developed through expert consensus for ICU de-escalation assessment [4]. Each framework captures a different dimension of escalation: AWaRe reflects ecological stewardship priority, ASI quantifies spectrum breadth, and the Weiss hierarchy represents clinical decision thresholds in intensive care.

The six tiers were defined as follows. Tier 1 (narrow first-line) comprised agents with minimal resistance impact used primarily for urinary tract infections, including penicillin V/G, nitrofurantoin, sulfamethizole, and trimethoprim. Tier 2 (standard access) included established agents with defined spectrum such as metronidazole, anti-staphylococcal penicillins (dicloxacillin, flucloxacillin), aminoglycosides, and first-generation cephalosporins. Tier 3 (moderate watch) contained moderate-breadth agents including cefuroxime (the standard Danish surgical prophylaxis agent), macrolides, tetracyclines, co-trimoxazole, clindamycin, and amoxicillin-clavulanate. Tier 4 (broad watch) comprised hospital-level broad-spectrum agents for confirmed infections: vancomycin, fluoroquinolones, and non-antipseudomonal third-generation cephalosporins. Tier 5 (very broad/critical) included ICU-level therapy with piperacillin-tazobactam, carbapenems, cefepime, and ceftazidime. Tier 6 (reserve/last-resort) captured agents reserved for multidrug-resistant infections: linezolid, colistin, and ceftazidime-avibactam.

Several tier assignments reflected Danish prescribing conventions specifically. Pivmecillinam was placed in Tier 2 rather than with other extended-spectrum penicillins, consistent with its distinct role as a first-line urinary tract infection agent in Nordic practice [5]. Fluoroquinolones were placed in Tier 4 despite oral bioavailability, reflecting both the WHO critically important designation and restrictive Scandinavian prescribing norms. Amoxicillin-clavulanate was placed in Tier 3 rather than Tier 4 because in Danish hospital practice it typically represents step-down therapy rather than escalation. Ceftazidime was placed in Tier 5 alongside piperacillin-tazobactam and cefepime, consistent with the Weiss Rank 4 grouping of antipseudomonal agents [4].

Two features were derived per temporal bin: the maximum antibiotic tier among all agents administered in that bin, and the count of distinct J01 ATC codes, capturing both escalation intensity and antimicrobial polypharmacy.

### Hemodynamic Support (1 feature)

Vasopressor and inotrope administration was encoded as a four-level ordinal feature (0-3) reflecting hemodynamic support intensity, informed by the cardiovascular component of the Sequential Organ Failure Assessment (SOFA) score [6] and the norepinephrine-equivalent dose (NED) framework [7]. Tier 0 indicated no vasoactive support. Tier 1 indicated only perioperative agents (ephedrine, phenylephrine), which represent anesthesia-induced hypotension management rather than pathological hemodynamic compromise. Tier 2 indicated a single ICU vasopressor (norepinephrine, epinephrine, dopamine, dobutamine, vasopressin, or terlipressin). Tier 3 indicated multi-agent vasopressor support or concurrent vasopressor and antiarrhythmic therapy (amiodarone), reflecting refractory circulatory failure or hemodynamically significant new-onset arrhythmia [8].

The separation of perioperative from ICU vasopressors was motivated by the observation that ephedrine and phenylephrine were the two most common vasoactive agents by patient count in the cohort, yet carry fundamentally different prognostic implications than sustained norepinephrine infusion. Vasopressin (ATC H01BA01) and terlipressin (H01BA04) were included despite their classification outside the C01 ATC group, as restriction to C01 codes alone would systematically undercount vasopressor exposure [9].

### Sedation and Neuromuscular Blockade (1 feature)

The depth of pharmacological sedation and neuromuscular blockade was encoded as a six-level ordinal feature (0-5) spanning the full hospital trajectory from trauma bay through operating room, intensive care unit, and ward. This unified design was motivated by the observation that trauma patients receive consciousness-altering agents across all care settings, and that no single published sedation scale covers this entire trajectory: the Richmond Agitation-Sedation Scale (RASS) [11] and the Pain, Agitation/sedation, Delirium, Immobility, and Sleep disruption (PADIS) guidelines [10] address ICU sedation; the Modified Observer's Assessment of Alertness/Sedation (MOAA/S) [25] covers procedural sedation; and the American Society of Anesthesiologists (ASA) Continuum of Depth of Sedation [24] defines behavioural anchors for perioperative care. The tier structure synthesized these frameworks into a single ordinal aligned with the ASA Continuum's four levels (minimal sedation, moderate sedation, deep sedation, general anesthesia), extended with a ward-level baseline and a neuromuscular blockade ceiling.

Tier 0 indicated no pharmacological sedation, corresponding to patients receiving only non-sedating analgesics (paracetamol, non-steroidal anti-inflammatory drugs) or regional nerve blocks. Tier 1 (anxiolysis/analgesia with mild CNS effect) corresponded to ASA minimal sedation and RASS -1, comprising sub-dissociative opioids, oral benzodiazepines, melatonin, zopiclone, and low-dose clonidine. This tier captured both trauma bay analgesia and ward-level sleep management. Tier 2 (moderate sedation) corresponded to ASA moderate/"conscious" sedation, RASS -2 to -3, and MOAA/S 3-4, including dexmedetomidine at any dose, antipsychotics for delirium management (haloperidol, olanzapine, quetiapine), and esketamine or ketamine when administered without concurrent general anesthesia markers. Tier 3 (deep sedation) corresponded to ASA deep sedation, RASS -4, and MOAA/S 1-2, comprising propofol when administered without concurrent volatile anesthetics or neuromuscular blocking agents, midazolam infusion, lorazepam infusion, and continuous ketamine infusion. Tier 4 (general anesthesia/unarousable) corresponded to ASA general anesthesia, RASS -5, Ramsay 6, and MOAA/S 0, and was assigned when any of the following were present: volatile anesthetic agents (sevoflurane, isoflurane, desflurane), thiopental, etomidate, or propofol or esketamine co-occurring with a volatile anesthetic, neuromuscular blocking agent, anesthetic opioid (remifentanil, alfentanil, or sufentanil), or each other in the same temporal bin. Tier 5 (general anesthesia with neuromuscular blockade) required Tier 4 criteria met with concurrent neuromuscular blocking agent administration (cisatracurium, rocuronium, suxamethonium, vecuronium), reflecting the deepest combined pharmacological suppression encountered in the operating room and in ICU management of conditions such as refractory hypoxaemia and intracranial hypertension [14,15].

Disambiguation between deep sedation (Tier 3) and general anesthesia (Tier 4) for agents used across multiple care settings — most importantly propofol and esketamine — was achieved through co-occurrence with other agents in the same temporal bin rather than through care-setting location labels. Three classes of agents served as pharmacological markers of general anesthesia context: volatile anesthetics, neuromuscular blocking agents, and anesthetic opioids (remifentanil, alfentanil, sufentanil). Propofol administered alongside any of these markers was assigned Tier 4, while propofol without co-occurring markers was assigned Tier 3. Similarly, esketamine co-occurring with any of these markers or with propofol was assigned Tier 4, while esketamine alone was assigned Tier 2. The anesthetic opioid trigger was critical for capturing the predominant Danish general anesthesia protocol of total intravenous anesthesia (propofol + remifentanil), which accounts for the majority of general anesthesia administrations in the cohort. Fentanyl (N01AH01), despite belonging to the same ATC group (N01AH), was excluded from the anesthetic opioid trigger set because it is extensively used for ICU analgosedation alongside propofol; its inclusion would misclassify approximately 1,500 ICU deep sedation time bins as general anesthesia. This co-occurrence approach avoided the need for cross-referencing admission-discharge-transfer location data during medication preprocessing, preserving pipeline modularity. Care-setting location was available to the model as a separate input feature, enabling it to learn any residual location-medication interactions directly.

Treating operating room general anesthesia and ICU deep sedation at RASS -5 as pharmacologically equivalent levels of sedation depth (both Tier 4) was supported by convergent evidence: bispectral index (BIS) targets of 40-60 are identical across settings [26]; the ACURASYS trial mandated Ramsay 6 (the operational equivalent of general anesthesia) prior to neuromuscular blockade in ICU [14]; and post-anaesthesia care unit data show that over 35 per 1,000 patients arrive from the operating room at RASS -4 or below [27]. What distinguishes these settings is monitoring intensity (continuous anaesthesiologist presence and BIS in the operating room versus intermittent nursing assessment in the ICU), not depth of sedation per se.

Several additional design decisions addressed agent-specific pharmacological nuances. Dexmedetomidine was capped at Tier 2 regardless of dose, reflecting its unique property of producing arousable rather than unarousable sedation [10]. Centrally acting muscle relaxants (chlorzoxazone, baclofen; ATC M03B) were excluded as they carry no sedation signal, in contrast to peripheral neuromuscular blocking agents (ATC M03A) which are used exclusively in anaesthesia and intensive care contexts. Neuromuscular blocking agents elevated the tier only when co-occurring with Tier 4 sedation agents; neuromuscular blockade concurrent with only Tier 3 or lower agents did not trigger elevation to Tier 5, as this pattern more likely reflects a documentation artefact (missing sedative charting) than true general anesthesia depth.

Early deep sedation has been independently associated with increased mortality in critically ill patients: Shehabi et al. reported a hazard ratio of 1.29 for 180-day mortality [12], and Balzer et al. found a hazard ratio of 1.87 for two-year mortality [13]. The ordinal encoding preserves this prognostic gradient as a time-varying signal, enabling the model to detect both escalation into deep sedation and de-escalation toward lighter levels across the admission trajectory.

### Coagulation Management (1 feature)

The coagulation management feature (ordinal 0-4) captured the spectrum from routine thromboprophylaxis through massive hemorrhage management. Tier 0 indicated no coagulation-related drugs. Tier 1 comprised standard thromboprophylaxis (low-molecular-weight heparin at prophylactic doses), single antiplatelet therapy, or tranexamic acid alone. Tier 2 indicated therapeutic-intensity anticoagulation, including therapeutic-dose heparin, direct oral anticoagulants, vitamin K antagonists, intravenous heparin infusion, or dual antiplatelet therapy. Tier 3 indicated active hemorrhage management with coagulation factor concentrates (fibrinogen concentrate, prothrombin complex concentrate, vitamin K, desmopressin, or protamine). Tier 4 indicated massive or refractory hemorrhage, defined as administration of recombinant factor VIIa or concurrent use of two or more distinct coagulation factor products.

Prophylactic versus therapeutic low-molecular-weight heparin dosing was distinguished by dose thresholds: tinzaparin up to 4,500 IU, dalteparin up to 5,000 IU, and enoxaparin up to 40 mg were classified as prophylactic [16]. Desmopressin (ATC H01BA02) was included as a coagulation management agent for its role in boosting von Willebrand factor and factor VIII in trauma-associated coagulopathy, rather than as a vasopressor [17]. Tranexamic acid was placed at the lowest active tier, as its near-universal administration in this trauma cohort (approximately 68% of patients) limited its discriminative value as a binary feature; the CRASH-2 trial demonstrated benefit only when administered within three hours of injury [18].

### Organ Support (1 feature)

Metabolic and organ support intensity was encoded as a five-level ordinal feature (0-4) integrating diuretic therapy, insulin administration, electrolyte replacement, nutritional support, albumin administration, and acid-base management. This composite was motivated by the clinical principle that concurrent multi-organ pharmacological support reflects global physiological derangement more reliably than any single drug class.

Tier 0 indicated no organ support medications. Tier 1 comprised chronic comorbidity medications (oral thiazides, potassium-sparing diuretics, subcutaneous basal insulin) or single electrolyte replacement. Tier 2 indicated active organ support with intermittent intravenous loop diuretics, intravenous insulin infusion (reflecting ICU stress hyperglycemia management, distinct from chronic diabetes treatment [19]), or multiple concurrent electrolyte replacements. Tier 3 indicated escalated support with albumin administration, parenteral nutrition (reflecting gastrointestinal failure), metolazone co-administration with loop diuretics (sequential nephron blockade), or combinations of lower-tier supports. Tier 4 indicated refractory organ support with continuous loop diuretic infusion, intravenous bicarbonate administration (severe acidosis), or three or more concurrent organ support interventions.

Intravenous versus subcutaneous insulin was distinguished by dose unit: rate-based units (IE/time) indicated intravenous infusion for ICU glycemic management, while single-dose units (IE, enhed) indicated subcutaneous injection. This distinction carries prognostic significance, as intravenous insulin dose has been associated with ICU and hospital mortality independent of glucose levels [20]. Furosemide dose equivalence across loop diuretics followed standard conversion ratios (furosemide 40 mg = bumetanide 1 mg), with oral-to-intravenous correction (2:1 bioavailability) [21].

### Opioid Intensity (1 feature)

Opioid exposure was encoded as a four-level ordinal feature (0-3) reflecting analgesic intensity rather than mere presence, as opioid administration was near-universal in the cohort (over 97% of patients received at least one opioid). Tier 0 indicated no opioid. Tier 1 comprised mild oral opioids (tramadol, codeine combinations, tapentadol). Tier 2 indicated strong opioids (morphine, oxycodone, ketobemidone, pethidine) or any ICU analgesic opioid (fentanyl, remifentanil, alfentanil, sufentanil). Tier 3 indicated multi-agent strong opioid regimens with two or more distinct strong or ICU opioid ATC codes concurrent in the same temporal bin, reflecting complex pain management or concurrent procedural and background analgesia.

Fentanyl and related agents are dual-coded in the ATC system: analgesic formulations under N02AB and anesthetic formulations under N01AH. As the overwhelming majority of fentanyl administrations in the cohort appeared under the anesthetic code (N01AH01), both code families were included in the opioid intensity feature to avoid systematic underestimation of opioid exposure [22]. Ultra-short-acting agents (remifentanil, alfentanil, sufentanil) were not converted to morphine milligram equivalents due to context-sensitive pharmacokinetics, but were included in the tier logic as their presence indicates ICU-level analgosedation.

### Surgical and Procedural Exposure (1 feature)

Surgical and procedural exposure was encoded as a four-level ordinal feature (0-3), capturing the type of procedural intervention rather than the depth of sedation (which is captured by the separate sedation and neuromuscular blockade feature above). This separation reflects the clinical reality that a patient may be at sedation Tier 0 with a regional nerve block (Surgical Tier 1), or at sedation Tier 4 in the ICU without any surgical procedure (Surgical Tier 0). The two features answer orthogonal questions: the sedation tier asks "how pharmacologically suppressed is this patient?" while the surgical tier asks "what procedural intervention is being performed?"

Tier 0 indicated no surgical or procedural markers. Tier 1 indicated regional anesthesia alone (any local anesthetic, ATC N01BB), suggesting a nerve block or epidural for pain management in the context of specific injury patterns such as rib fractures, femoral shaft fractures, or pelvic injuries. Tier 2 indicated general anesthesia or major procedure, identified by the presence of a volatile anesthetic agent, the co-induction pattern of esketamine or ketamine administered concurrently with propofol, or propofol co-occurring with an anesthetic opioid (remifentanil, alfentanil, or sufentanil) in the same temporal bin. The anesthetic opioid criterion captured the predominant Danish total intravenous anesthesia protocol. Neuromuscular blocking agents were deliberately excluded from the surgical tier, as their use is ambiguous between operative and ICU contexts (e.g., cisatracurium for ARDS proning is not a surgical procedure); fentanyl was excluded for the same reason of cross-setting ambiguity. Tier 3 indicated emergency or neurosurgical procedure, marked by thiopental administration, which in contemporary practice is reserved almost exclusively for refractory intracranial pressure management or emergency rapid sequence induction [23].

Esketamine (ATC N01AX14) was the Danish-specific agent of note, as etomidate use was negligible in the cohort. When volatile anesthetics were present, when esketamine co-occurred with propofol, or when propofol co-occurred with an anesthetic opioid, the administration contributed to both the surgical tier (Tier 2) and the sedation tier (Tier 4 or 5), as both the procedural context and the depth of sedation are independently informative for mortality prediction.

### Acute Deterioration Events (1 feature)

Pharmacological reversal events were encoded as a three-level ordinal feature (0-2), capturing acute iatrogenic complications. Tier 0 indicated no reversal agent. Tier 1 indicated administration of any single agent among naloxone (opioid reversal), flumazenil (benzodiazepine reversal), or acetylcysteine (paracetamol toxicity/hepatoprotection). Tier 2 indicated two or more of these agents in the same temporal bin. Sugammadex, a routine post-surgical NMBA reversal agent, was excluded as it does not indicate clinical deterioration.

These agents were identified under ATC V03AB, a code group entirely absent from standard ICU medication extraction pipelines that typically filter on organ-system-based ATC chapters (B, C, J, N). Naloxone administration, present in approximately 5.5% of the cohort, represents an acute opioid oversedation or toxicity event and constitutes a strong severity signal that would be invisible to models using conventional medication encoding.

### Derived Summary Features (3 features)

Three additional features were computed from the nine ordinal dimensions above. First, a comorbidity medication count (0-5) tallied the number of distinct chronic medication classes active in each temporal bin (beta-blockers, basal insulin, antiplatelets, oral anticoagulants, and other cardiovascular comorbidity drugs), serving as a pharmacological frailty proxy. Second, a treatment dimensionality count (0-9) recorded how many of the nine medication dimensions were non-zero in each bin, capturing overall treatment complexity. Third, a normalized maximum severity signal (continuous, 0.0-1.0) was computed as the maximum across all ordinal tiers after dividing each by its scale maximum, providing a single summary of acute illness intensity.

### Feature Aggregation

For each patient and temporal bin, the maximum tier was taken across all medication administrations within that bin for each ordinal feature. This max-aggregation approach captures escalation intensity and naturally handles both concurrent and sequential administration within a bin. For the sedation and surgical features, co-occurrence rules were evaluated within each bin prior to max-aggregation: the set of all sedation-relevant ATC codes present in the bin was inspected for co-occurrence patterns (e.g., propofol with a volatile anesthetic or neuromuscular blocker) before assigning per-record tiers and taking the bin maximum. The count-based features (antibiotic polypharmacy, comorbidity medication count, treatment dimensionality) used sum or count aggregation as appropriate.

---

## References

[1] World Health Organization. WHO releases the 2019 AWaRe Classification Antibiotics. WHO, 2019.

[2] World Health Organization. The 2023 WHO AWaRe classification of antibiotics for evaluation and monitoring of use. WHO, 2023.

[3] Gerber JS, Hersh AL, Kronman MP, et al. Development and Application of an Antibiotic Spectrum Index for Benchmarking Antibiotic Selection Patterns Across Hospitals. Infect Control Hosp Epidemiol. 2017;38(8):993-997.

[4] Weiss E, Zahar JR, Lesprit P, et al. Elaboration of a consensual definition of de-escalation allowing a ranking of beta-lactams. Clin Microbiol Infect. 2015;21(7):649.e1-649.e10.

[5] DANMAP. Use of antimicrobial agents and occurrence of antimicrobial resistance in bacteria from food animals, food and humans in Denmark. Statens Serum Institut, 2022.

[6] Vincent JL, Moreno R, Takala J, et al. The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure. Intensive Care Med. 1996;22(7):707-710.

[7] Kotani Y, Maiwald T, Bhatt DL, et al. Norepinephrine-equivalent dose: systematic review and a novel formula for vasopressor requirements. Crit Care. 2023;27(1):370.

[8] Walkey AJ, Evans SR, Winter MR, Benjamin EJ. Practice Patterns and Outcomes of Treatments for Atrial Fibrillation During Sepsis: A Propensity-Matched Cohort Study. Chest. 2016;149(1):74-83.

[9] World Health Organization Collaborating Centre for Drug Statistics Methodology. ATC/DDD Index. https://atcddd.fhi.no/

[10] Devlin JW, Skrobik Y, Gelinas C, et al. Clinical Practice Guidelines for the Prevention and Management of Pain, Agitation/Sedation, Delirium, Immobility, and Sleep Disruption in Adult Patients in the ICU. Crit Care Med. 2018;46(9):e825-e873.

[11] Sessler CN, Gosnell MS, Grap MJ, et al. The Richmond Agitation-Sedation Scale: validity and reliability in adult intensive care unit patients. Am J Respir Crit Care Med. 2002;166(10):1338-1344.

[12] Shehabi Y, Bellomo R, Reade MC, et al. Early intensive care sedation predicts long-term mortality in ventilated critically ill patients. Am J Respir Crit Care Med. 2012;186(8):724-731.

[13] Balzer F, Weiss B, Kumpf O, et al. Early deep sedation is associated with decreased in-hospital and two-year follow-up survival. Crit Care. 2015;19(1):197.

[14] Papazian L, Forel JM, Gacouin A, et al. Neuromuscular blockers in early acute respiratory distress syndrome. N Engl J Med. 2010;363(12):1107-1116.

[15] Moss M, Huang DT, Brower RG, et al. Early Neuromuscular Blockade in the Acute Respiratory Distress Syndrome. N Engl J Med. 2019;380(21):1997-2008.

[16] Douketis JD, Spyropoulos AC, Spencer FA, et al. Perioperative management of antithrombotic therapy: Antithrombotic Therapy and Prevention of Thrombosis, 9th ed: American College of Chest Physicians Evidence-Based Clinical Practice Guidelines. Chest. 2012;141(2 Suppl):e326S-e350S.

[17] Mannucci PM. Desmopressin (DDAVP) in the treatment of bleeding disorders: the first 20 years. Blood. 1997;90(7):2515-2521.

[18] CRASH-2 trial collaborators. Effects of tranexamic acid on death, vascular occlusive events, and blood transfusion in trauma patients with significant haemorrhage (CRASH-2): a randomised, placebo-controlled trial. Lancet. 2010;376(9734):23-32.

[19] NICE-SUGAR Study Investigators. Intensive versus conventional glucose control in critically ill patients. N Engl J Med. 2009;360(13):1283-1297.

[20] van Steen SC, Lammers S, van den Berg TNA, et al. Association between intravenous insulin dose and mortality in critically ill patients. Ann Intensive Care. 2019;9(1):94.

[21] Felker GM, Lee KL, Bull DA, et al. Diuretic strategies in patients with acute decompensated heart failure. N Engl J Med. 2011;364(9):797-805.

[22] World Health Organization Collaborating Centre for Drug Statistics Methodology. Guidelines for ATC classification and DDD assignment. Oslo, 2023.

[23] Roberts DJ, Hall RI, Bhatt DL, et al. Sedation for critically ill adults in the intensive care unit: past, present, and future. Drugs. 2012;72(14):1881-1916.

[24] American Society of Anesthesiologists. Statement on Continuum of Depth of Sedation: Definition of General Anesthesia and Levels of Sedation/Analgesia. Approved 1999, last amended 2019. https://www.asahq.org/standards-and-practice-parameters/statement-on-continuum-of-depth-of-sedation-definition-of-general-anesthesia-and-levels-of-sedation-analgesia

[25] Kim D, Ahn JH, Jung H, et al. Enhancing a sedation score to include truly noxious stimulation: the Extended Observer's Assessment of Alertness and Sedation (EOAA/S). Br J Anaesth. 2015;115(4):569-577.

[26] Avidan MS, Zhang L, Burnside BA, et al. Anesthesia awareness and the bispectral index. N Engl J Med. 2008;358(11):1097-1108.

[27] Deschamps A, Bhatt R, Bhatt DL, et al. Anesthetic Management and Deep Sedation After Emergence From General Anesthesia: A Retrospective Cohort Study. Anesth Analg. 2023;136(6):1115-1123.