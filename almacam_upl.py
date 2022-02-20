import scaleapi

client = scaleapi.ScaleClient("test_5c8ba3db81bd443e95f6201e245b403f")


from scaleapi.tasks import TaskType
from scaleapi.exceptions import ScaleDuplicateResource

payload = dict(
    project = "test_project",
    instruction = '''
    Please draw bounding boxes around people, animals, and other items in this task. This is from a live, public webcam on the UIUC Campus. The environment should be relatively static. Each task will contain a random sample of 10 images from the dataset. 
Please make sure to draw bounding boxes around all individuals in the image. 
If they are partially occluded, please draw the bounding box around the entire non-occluded region. 
Please keep an eye out for individuals in the background.
''',
    attachment_type = "image",
    attachment = "http://i.imgur.com/v4cBreD.jpg",
    unique_id = "c235d023af73",
    geometries = {
        "box": {
            "objects_to_annotate": ["Baby Cow", "Big Cow"],
            "min_height": 10,
            "min_width": 10,
        }
    },
)

try:
    client.create_task(TaskType.ImageAnnotation, **payload)
except ScaleDuplicateResource as err:
    print(err.message)  # If unique_id is already used for a different task