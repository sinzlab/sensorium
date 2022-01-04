import datajoint as dj
import os

# enable support for native Python datatypes in blobs
dj.config["enable_python_native_blobs"] = True

if not "stores" in dj.config:
    dj.config["stores"] = {}
dj.config["stores"] = {
    "nnfabrik": {  #  store in s3
        "protocol": "s3",
        "endpoint": os.environ.get("S3_ENDPOINT", "FAKE_ENDPOINT"),
        "bucket": "nnexplore",
        "location": "dj-store",
        "access_key": os.environ.get("S3_ACCESS_KEY", "FAKEKEY"),
        "secret_key": os.environ.get("S3_SECRET_KEY", "FAKEKEY"),
    }
}
