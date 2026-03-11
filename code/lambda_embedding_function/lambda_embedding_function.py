import boto3
import csv
import json
import io
import os


INPUT_BUCKET = "customer-embedding-input-bucket"
INPUT_KEY    = "customers_demo.csv"                  

OUTPUT_BUCKET = "customer-embedding-output-bucket"
OUTPUT_KEY    = "customers_with_embeddings.csv" 

MODEL_ID = "amazon.titan-embed-text-v2:0"



s3 = boto3.client("s3")
bedrock = boto3.client("bedrock-runtime")


def generate_embedding(text: str):
    payload = {
        "inputText": text
    }

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload)
    )

    resp_body = json.loads(response["body"].read())
    return resp_body["embedding"]


def lambda_handler(event, context):

    obj = s3.get_object(Bucket=INPUT_BUCKET, Key=INPUT_KEY)
    csv_raw = obj["Body"].read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(csv_raw))

    # Prepare output CSV
    output_buf = io.StringIO()
    fieldnames = reader.fieldnames + ["embedding"]
    writer = csv.DictWriter(output_buf, fieldnames=fieldnames)
    writer.writeheader()


    for row in reader:
        text = row.get("search_text", "")

        embedding = generate_embedding(text)

        row["embedding"] = json.dumps(embedding)
        writer.writerow(row)


    s3.put_object(
        Bucket=OUTPUT_BUCKET,
        Key=OUTPUT_KEY,
        Body=output_buf.getvalue().encode("utf-8"),
        ContentType="text/csv"
    )

    output_uri = f"s3://{OUTPUT_BUCKET}/{OUTPUT_KEY}"

    return {
        "statusCode": 200,
        "message": "Embeddings generated and uploaded!",
        "output_s3": output_uri
    }
