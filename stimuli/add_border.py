from PIL import Image

savepath = "withborder/"
colors = ["#FF0000", "#2E8B57", "#0000FF"]
conditions = ["f", "u", "s"]
index = 0
for condition in conditions:
    for i in range(150):
        current_img = Image.open("stimuli_"+condition+str(i+1).zfill(3)+".bmp")
        x, y = current_img.size
        print(x, y)
        new_pic = Image.new('RGB', (x + 12, y + 12), color=colors[index])
        new_pic.paste(current_img, (6, 6))
        new_pic.save(savepath + condition+str(i+1).zfill(3) + ".jpg")
    index = index + 1