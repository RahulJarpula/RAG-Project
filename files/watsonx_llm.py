from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import WatsonxLLM

def get_llm():
    model_id = 'ibm/granite-3-3-8b-instruct'
    parameters = {
        GenParams.MAX_NEW_TOKENS: 1024,
        GenParams.TEMPERATURE: 0.5,
    }
    return WatsonxLLM(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id="9c60d3de-cd1c-4211-9d56-52a22f83ccbb",
        apikey = "OPg1pgO41pmJ5odX8Fyh8u_h0hal1C9Ikx6i5HGbdBf1",
        params=parameters,
    )
