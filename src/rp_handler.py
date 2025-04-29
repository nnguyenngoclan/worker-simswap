import swap_video
import runpod
import os
import httpx
import asyncio
from swap_video import swap_video
import uuid
from google.cloud import storage

def upload_file_to_gcs(file_path, destination_blob_name):
    service_account_json = "/upload_file_key.json"
    bucket_name = "swap_video"
    # Initialize the storage client
    storage_client = storage.Client.from_service_account_json(service_account_json)

    bucket = storage_client.bucket(bucket_name, user_project="melofi-e4f24")
    blob = bucket.blob(destination_blob_name)

    # Upload the file
    blob.upload_from_filename(file_path)

    # Make the blob publicly viewable (optional)
    blob.make_public()

    public_url = blob.public_url
    return public_url

async def download_file(client, key, url):
    if url is None:
        return key, None
        
    try:
        filename = url.split("/")[-1] or "downloaded_file"
        file_path = os.path.join("downloads", filename)
        response = await client.get(url)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)

            print(f"Downloaded {filename}")
            return key, file_path  # 🔥 Return the key and the saved path
        else:
            print(f"Download file failed: {response.status_code} - {url}")
            raise ValueError(f"Download failed with status: {response.status_code}")
    except Exception as e:
        raise ValueError(f"Download failed for {url}") from e
        
async def run(job):
    try:
        job_input = job['input']
        task = job_input["task"]
        source_image_url = job_input["source_image"]
        target_image_url = job_input["target_image"]
        video_url = job_input.get("video")
        image_url = job_input.get("image")

        os.makedirs("downloads", exist_ok=True)

        url_map = {
            "source_image": source_image_url,
            "target_image": target_image_url,
            "video": video_url, 
            "image": image_url
        }

        # Open an async HTTP client session
        async with httpx.AsyncClient() as client:
            tasks = [download_file(client, key, url) for key, url in url_map.items()]
            results = await asyncio.gather(*tasks)

            # 🔥 Build a mapping: { key -> downloaded file path }
            downloaded_files = {key: path for key, path in results}
            print("Download file succeed: ", downloaded_files)

            source_image_path = downloaded_files["source_image"]
            target_image_path = downloaded_files["target_image"]
            output_path = None
            unique_id = uuid.uuid4().hex  # unique file name
            if task == "video":
                video_path = downloaded_files["video"]                
                output_path = os.path.join("output", f"swapped_{unique_id}.mp4")                        
                swap_video(source_image_path, target_image_path, video_path, output_path)
            elif task == "image":
                from test_wholeimage_swapspecific import swap_image
                image_path = downloaded_files["image"]
                output_path = os.path.join("output", f"swapped_{unique_id}.png")                  
                swap_image(source_image_path, target_image_path, image_path, output_path)
            else:
                raise ValueError(f"Unsupported type value: {task}")
            
            # Upload to Google Cloud Storage                        
            destination_blob_name = os.path.basename(output_path)
            file_url = upload_file_to_gcs(output_path, destination_blob_name)
            print("Upload file succeeded: ", file_url)

            return {"status": "ok", "file_url": file_url}
    except Exception as e:
        error = str(e)
        print(f"Run job failed: {error}")
        return {"status": f"failed: {error}"}
    

if __name__ == "__main__":
    runpod.serverless.start({"handler": run})


