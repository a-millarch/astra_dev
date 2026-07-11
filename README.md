# ASTRA - AI for Surgical Trauma Risk Assessment
<i>Center for Surgical Translational and Artificial Intelligence Research (CSTAR), Copenhagen University Hospital, Denmark</i>

As a part of the ASTRA project by CSTAR, an ML-driven risk assessment tool for trauma patients is developed for implementation in the Eletronic Health Record (EHR) system at the Copenhagen University Hospital. 

> **Implementing ASTRA for inference?** Start with [docs/HANDOFF.md](docs/HANDOFF.md) — it covers the `AstraPredictor` Python API, the FastAPI reference service, the artifact bundle (`python -m astra.inference.export_artifacts`), and the normative per-concept data contract for wiring up your own EHR data feed. Then run `python scripts/demo_api_usage.py` — a self-contained end-to-end demo (synthetic model + synthetic patient, no artifacts or data needed).
>
> **Operating the full lifecycle** (new data dump → rebuild → retrain → evaluate → redeploy): [docs/RETRAINING.md](docs/RETRAINING.md).

### Single patient continious update process graph
```mermaid
---
config:
  layout: dagre
  theme: neo-dark
  look: neo
---
flowchart TB
 subgraph MainPipeline["<b>Pipeline Loop</b><br>(Periodic Update)"]
        BuildPatientInfo["build_patient_info"]
        BaseDF["base_df"]
        HistoricBaseDF[("data/interim/historic_base_df")]
        subgraph PreProces["Model-specific<br>pre-processing"]
          ConstructDatasets["<b>construct_datasets</b><br>update tabular, update aggregation into bin_df"]
          TabularDS[/"`tabular_ds`"/]
          TimeseriesDS[/"`timeseries_ds`"/]
          Dataloaders["DataLoaders<br>(mixed)"]
        end
        Predict["predict"]
        PredictionsDF[("data/output/predictions.csv")]
  end
    BuildPatientInfo --> BaseDF & BinDF["bin_df"]
    HistoricFeatures["compute_historic_features"] <-- run once --> BuildPatientInfo
    BaseDF --> HistoricBaseDF & ConstructDatasets
    BinDF --> ConstructDatasets
    ConstructDatasets --> TabularDS & TimeseriesDS
    TabularDS --> Dataloaders
    TimeseriesDS --> Dataloaders
    Dataloaders --> Predict
    Predict --> PredictionsDF
    PredictionsDF -- wait interval --> BuildPatientInfo
    TraumaCall(["`<u>Trauma Call Trigger</u>`"]) -- from DAP ETL --> dbIdTraumaCallCSV[("data/external/trauma_call.csv")]
    dbIdTraumaCallCSV --> BuildPatientInfo
    EndTrajectory(["`End Trajectory Trigger<br>(Discharge/Death)`"]) -- update message --> BuildPatientInfo
    BuildPatientInfo -. final update .-> HistoricBaseDF
    PredictionsDF <--> GUI_API[["`GUI API end-point`"]]
```



### Historic data process graph
```mermaid
---
config:
  layout: dagre
  theme: neo-dark
  look: neo
---
flowchart TB
    A["Blobstore: SP-data"] -- Patient level filtering (Trauma call) --> B["data/raw"]
    B -- Constructing historical and static data --> C1["data/interim/base_df"]
    B -- "Temporal Filtering: In-hospital Data Filtering" --> C["data/interim/concepts"]
    C -- Mapping to bin_df by aggregation --> D["data/interim/mapped"]
    D <-- "Pre-proces and transform" --> E["TSDS Data Class"]
    E --> F["Tabular Dataset"] & G["Timeseries Dataset"]
    F --> F1["Tabular dataloader<br>(FastAI)"]
    G --> G1["Timeseries dataloader<br>(FastAI)"]
    F1 --> M["Mixed dataloader<br>(TSAI)"]
    G1 --> M
```


### Single patient lifecycle state diagram
```mermaid
stateDiagram-v2
    direction TB

    [*] --> AwaitingTraumaCall
    AwaitingTraumaCall : Awaiting Trauma Call

    AwaitingTraumaCall --> TraumaCallReceived : DAP ETL writes trauma_call.csv

    TraumaCallReceived : Trauma Call Received
    TraumaCallReceived --> PatientIngestion

    state PatientIngestion {
        direction TB

        state BuildBaseDF {
            direction LR
            load_pop : load_or_collect_population
            build_traj : build_trajectories
            match_traj : match_population_to_trajectories
            add_info : add_patient_info
            compute_hist : compute_historic_features
            load_pop --> build_traj
            build_traj --> match_traj
            match_traj --> add_info
            add_info --> compute_hist
        }

        BuildBaseDF --> BuildBinDF

        state BuildBinDF {
            direction LR
            apply_intervals : apply bin_intervals config
            gen_bins : generate time bins
            save_bins : save bin_df
            apply_intervals --> gen_bins
            gen_bins --> save_bins
        }
    }

    PatientIngestion --> MonitoringLoop

    state MonitoringLoop {
        direction TB

        [*] --> PreProcessing
        PreProcessing : Update tabular features and aggregate into bin_df

        PreProcessing --> ConstructDatasets

        state ConstructDatasets {
            direction LR
            tabular_ds : tabular_ds
            timeseries_ds : timeseries_ds
            mixed_dl : Mixed DataLoaders
            tabular_ds --> mixed_dl
            timeseries_ds --> mixed_dl
        }

        ConstructDatasets --> Predict
        Predict : Model forward pass to predictions.csv

        Predict --> ServeAPI
        ServeAPI : GUI API endpoint serves prediction

        ServeAPI --> WaitInterval
        WaitInterval : Wait configured interval
    }

    state end_check <<choice>>

    MonitoringLoop --> end_check : check for end trigger

    end_check --> MonitoringLoop : patient still active
    end_check --> EndTrajectory : discharge or death

    state EndTrajectory {
        direction TB
        final_update : Final build_patient_info update
        final_predict : Final prediction cycle
        archive : Archive to historic_base_df

        final_update --> final_predict
        final_predict --> archive
    }

    EndTrajectory --> AwaitingTraumaCall : return to idle
```