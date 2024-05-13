import json
from PIL import Image, ImageDraw


def draw_strokes_on_image(stroke_data, image_path, filename):
    try:
        page_id = "55175"
        image_file = f"back.png"
        new_path = "out_pics/" + filename + ".png"
        img = Image.open(image_file)
        draw = ImageDraw.Draw(img)
        for strokes in stroke_data:
            for i in range(len(strokes) - 1):
                start_point = (strokes[i]["x"] * (11.810055), strokes[i]["y"] * (11.810055))
                end_point = (strokes[i + 1]["x"] * (11.810055), strokes[i + 1]["y"] * (11.810055))
                width = strokes[i]["linewidth"]
                draw.line([start_point, end_point], fill="black", width=int(width), joint="miter")
        img.save(new_path)

    except Exception as e:
        print(e)


FOLDER_PATH = "E:\\money\\money_python\\server\\json_data\\"
image_path = "E:\\money\\money_python\\hand_writing_demo2"


def draw(pth, ch):
    json_path = pth
    with open(json_path, "r") as f:
        stroke_data = json.load(f)["json"]
    draw_strokes_on_image(stroke_data, image_path, ch)


draw("train_sets/两/1.json", "两")
