# LLM Case Study for OD Matrix Calibration

This repository contains two main modules for calibrating a 56×56 Origin-Destination (OD) matrix using Large Language Models (LLMs):

1. **ExpertLLM Code** – Optimizes the OD matrix using an Expert LLM pipeline.
2. **LLM_OD_Link** – Maps OD pair contributions to link performance.

## Requirements

- Required Python packages:
  - numpy
  - pandas
  - torch
  - transformers
  - accelerate
  - python-dotenv
- **Wine64** (for running the simulation executable on Linux)
- Access to a **Hugging Face API Key**

## Configure Environment Variables:
- Create a .env file in the project root with your Hugging Face API key:
- ```HUGGING_FACE_API_KEY=your_huggingface_api_key_here```

## Running the Code
### ExpertLLM Code
To run the ExpertLLM pipeline from the root directory, run the script with:

``` python pipeline/llm_od_link.py ```

The code will use the ```.env``` for the Hugging Face API key and ```config/config.json``` for all paths.

### LLM_OD_Link Code
To run the OD Link mapping pipeline from the root directory, run the script with:

``` python pipeline/od_link.py```

The code will run simulations, create OD pair mappings, and output the results to the specified CSV files.