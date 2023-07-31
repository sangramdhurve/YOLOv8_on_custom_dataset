# train-yolov8-custom-dataset-step-by-step-guide


## dataset

If you want to train yolov8 with the same dataset I use in the video, this is what you should do:

1. Download the [downloader.py](https://raw.githubusercontent.com/openimages/dataset/master/downloader.py) file.
2. Download the object detection dataset; [train](https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv), [validation](https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv) and [test](https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv).
2. Go to **prepare_data** directory.
4. Execute **create_image_list_file.py**.
5. Execute **downloader.py**.

       python downloader.py $IMAGE_LIST_FILE --download_folder=$DOWNLOAD_FOLDER

6. Execute **create_dataset_yolo_format.py**, changing **DATA_ALL_DIR** by **$DOWNLOAD_FOLDER**.
# YOLOv8_on_custom_dataset
# YOLOv8_on_custom_dataset
