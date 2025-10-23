# Used Copilot/ChatGPT for assistance in coding & understanding

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 100)
        img_height: Height of the image (default: 150)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """
    # TODO
    kart_objs = []
    # Read info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Calculate image center
    img_center_x = img_width // 2
    img_center_y = img_height // 2

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    karts = info["karts"]
    frame_detections = info["detections"][view_index]

    # Extract karts from detections
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:  # Not a kart
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue
        # Skip if kart is out of sight
        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get kart name
        kart_name = karts[track_id]

        # Calculate kart's center
        kart_center_x = (x1_scaled + x2_scaled) // 2
        kart_center_y = (y1_scaled + y2_scaled) // 2

        kart_objs.append({
            "instance_id": track_id,
            "kart_name": kart_name,
            "center": (kart_center_x, kart_center_y),
        })
    
    # Find closest kart by Euclidean distance
    closest_kart = None
    closest_distance = float("inf")
    for kart in kart_objs:
        kart_center_x, kart_center_y = kart["center"]
        distance = np.sqrt((kart_center_x - img_center_x) ** 2 
                           + (kart_center_y - img_center_y) ** 2)
        if distance < closest_distance:
            closest_distance = distance
            closest_kart = kart
    # Add is_center_kart flag
    for kart in kart_objs:
        if kart == closest_kart:
            kart["is_center_kart"] = True
        else:
            kart["is_center_kart"] = False

    return kart_objs
    # raise NotImplementedError("Not implemented")

def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    # TODO
    with open(info_path) as f:
        info = json.load(f)
    # Extract track name
    track_name = info["track"]
    return track_name
    # raise NotImplementedError("Not implemented")


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 100)
        img_height: Height of the image (default: 150)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # Extract karts
    kart_objs = extract_kart_objects(info_path, view_index)
    # print(kart_objs)
    # Extract track name
    track_name = extract_track_info(info_path)
    # print(track_name)

    # Generate question-answer pairs
    qa_pairs = []

    # 1. Ego car question
    # What kart is the ego car?
    ego_car = None
    for kart_obj in kart_objs:
        if kart_obj["is_center_kart"]:
            ego_car = kart_obj["kart_name"]
            ego_center_x, ego_center_y = kart_obj["center"]
    if ego_car is None:
            qa_pairs.append({
            "question": "What kart is the ego car?",
            "answer": "unknown",
        })
    else:
        qa_pairs.append({
            "question": "What kart is the ego car?",
            "answer": ego_car,
        })

    # 2. Total karts question
    # How many karts are there in the scenario?
    qa_pairs.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(len(kart_objs)),
    })

    # 3. Track information questions
    # What track is this?
    qa_pairs.append({
        "question": "What track is this?",
        "answer": track_name,
    })

    # Stop generating if no ego car
    if ego_car == None:
        return qa_pairs

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    left_karts = 0
    right_karts = 0
    front_karts = 0
    back_karts = 0
    for kart_obj in kart_objs:
        if kart_obj["is_center_kart"]:
            continue
        kart_name = kart_obj["kart_name"]
        kart_center_x, kart_center_y = kart_obj["center"]
        kart_positions = []
        if kart_center_x < ego_center_x:
            left_karts += 1
            qa_pairs.append({
                "question": f"Is {kart_name} to the left or right of the ego car?",
                "answer": "left",
            })
            kart_positions.append("left")
        elif kart_center_x > ego_center_x:
            right_karts += 1
            qa_pairs.append({
                "question": f"Is {kart_name} to the left or right of the ego car?",
                "answer": "right",
            })
            kart_positions.append("right")
        else:
            qa_pairs.append({
                "question": f"Is {kart_name} to the left or right of the ego car?",
                "answer": "neither",
            })
        if kart_center_y < ego_center_y:
            front_karts += 1
            qa_pairs.append({
                "question": f"Is {kart_name} in front of or behind the ego car?",
                "answer": "front",
            })
            kart_positions.append("front")
        elif kart_center_y > ego_center_y:
            back_karts += 1
            qa_pairs.append({
                "question": f"Is {kart_name} in front of or behind the ego car?",
                "answer": "back",
            })
            kart_positions.append("back")
        else:
            qa_pairs.append({
                "question": f"Is {kart_name} in front of or behind the ego car?",
                "answer": "neither",
            })

        # Extra Q
        if len(kart_positions) == 2:
            qa_pairs.append({
                "question": f"Where is {kart_name} relative to the ego car?",
                "answer": f"{kart_positions[1]} and {kart_positions[0]}",
            })
        elif len(kart_positions) == 1:
            qa_pairs.append({
                "question": f"Where is {kart_name} relative to the ego car?",
                "answer": kart_positions[0],
            })
        else:
            continue
            
    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?
    qa_pairs.append({
        "question": "How many karts are to the left of the ego car?",
        "answer": str(left_karts),
    })
    qa_pairs.append({
        "question": "How many karts are to the right of the ego car?",
        "answer": str(right_karts),
    })
    qa_pairs.append({
        "question": "How many karts are in front of the ego car?",
        "answer": str(front_karts),
    })
    qa_pairs.append({
        "question": "How many karts are behind the ego car?",
        "answer": str(back_karts),
    })

    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


def generate_all_train_qa_pairs(outfile: str = 'my_qa_pairs.json'):
    from tqdm import tqdm
    path = Path(__file__).parent.parent / "data/train"
    outfile_path = path / outfile
    # Check if output file already exists
    if Path(outfile_path).exists():
        print(f"Output file {outfile} already exists. Please remove it first.")
        return
    # Create new output file
    with open(outfile_path, 'w') as file:
        file.write("[\n")  # Start an open list
        first_entry = True

        print(f"Generating all QA pairs for images in {path}...")
        # Find all info.json files
        json_files = list(path.glob("**/*_info.json"))
        # Generate and save QA pairs for each info.json file
        for json_file in tqdm(json_files):
            for view_index in range(10):
                image_file = f"{json_file.stem.split('_')[0]}_{view_index:02d}_im.jpg"
                qa_pairs = generate_qa_pairs(json_file, view_index)
                qa_pairs = [{"question": qa["question"], "answer": qa["answer"], 
                            "image_file": f"train/{image_file}"} for qa in qa_pairs]
                # Add QA pairs to output file
                for qa in qa_pairs:
                    if not first_entry:
                        file.write(",\n")
                    json.dump(qa, file, indent=4)
                    first_entry = False

        file.write("\n]\n")  # end list


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_qa_pairs, 
               "draw": draw_detections,
               "extract_karts": extract_kart_objects, 
               "extract_track": extract_track_info,
               "generate": generate_qa_pairs,
               "generate_all": generate_all_train_qa_pairs})


if __name__ == "__main__":
    main()
