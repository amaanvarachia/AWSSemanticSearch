import os
import json
import boto3
import requests
from requests_aws4auth import AWS4Auth


AOSS_ENDPOINT = os.environ["AOSS_ENDPOINT"]
AOSS_INDEX = os.environ["AOSS_INDEX_NAME"]
MODEL_ID = os.environ["BEDROCK_MODEL_ID"]


session = boto3.Session()
region = session.region_name or "us-east-1"
credentials = session.get_credentials()

aws_auth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    "aoss",
    session_token=credentials.token
)

bedrock = boto3.client("bedrock-runtime", region_name=region)


def embed_text(text):
    payload = { "inputText": text }

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload)
    )

    output = json.loads(response["body"].read())
    return output["embedding"]


def search_vector(vector, k=3):
    query = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": vector,
                    "k": k
                }
            }
        }
    }

    url = f"{AOSS_ENDPOINT}/{AOSS_INDEX}/_search"

    resp = requests.post(   # IMPORTANT: POST not GET
        url,
        auth=aws_auth,
        headers={"Content-Type": "application/json"},
        data=json.dumps(query)
    )

    return resp.json()


def lambda_handler(event, context):
    body = json.loads(event.get("body", "{}"))
    query_text = body.get("query")

    if not query_text:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Missing 'query' in body"})
        }

    
    vector = embed_text(query_text)

   
    results = search_vector(vector)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "query": query_text,
            "results": results
        })
    }
