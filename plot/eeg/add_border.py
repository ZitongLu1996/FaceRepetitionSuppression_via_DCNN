from PIL import Image

img_path = "demo_stimuli/"
imgs = ["f.bmp", "f.bmp", "f.bmp", "u.bmp", "u.bmp", "u.bmp", "s.bmp", "s.bmp", "s.bmp"]
colors = ["#FF0000", "#FF6347", "#FFA07A", "#2E8B57", "#32CD32", "#98FB98", "#0000FF", "#6495ED", "#87CEFA"]
newimgs = ["fn", "fe", "fl", "un", "ue", "ul", "sn", "se", "sl"]

for i in range(9):
    current_img = Image.open(img_path + imgs[i])
    x, y = current_img.size
    print(x, y)
    new_pic = Image.new('RGB', (x + 18, y + 18), color=colors[i])
    new_pic.paste(current_img, (9, 9))
    new_pic.save(img_path + newimgs[i] + ".jpg")