import cv2 as cv
import fitz
import io
import numpy as np
import os
from PIL import Image
import re
import sys

date_re = r'[0-3][0-9]\.[01][0-9][.,]\.?[12][0-9][0-9][0-9]'
TEMP_IMG_NAME = 'tempimg'
TEMP_PAGE_NAME = 'temppage.jpg'
IMAGES_FOLDER = 'images'

def open_cv_reduce(page, imgs):
    reduced = []
    for img in imgs:
        xref = img[0]
        base_image = pdf_file.extractImage(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        image = Image.open(io.BytesIO(image_bytes))
        (max_match, max_loc) = (0.0, None)
        common_pairs = [(1.0, 1.0), (1.0, 1.05), (0.9, 1.05), (0.9, 1.1), (None, None), (0.9, 1.2)]
        for w, h in common_pairs:
            #print((w,h))
            bbox = page.getImageBbox(img).round()
            if bbox.width < 10 or bbox.height < 10:
                continue

            if w == None and h == None:
                from PIL import ImageOps
                current = ImageOps.mirror(image.copy()).resize((bbox.width, bbox.height))
            else:
                bbox[2] = round(bbox[2] * w)
                bbox[1] = round(bbox[1] * h)
                try:
                    current = image.copy().resize((bbox.width, bbox.height))
                except ValueError:
                    continue
            pix = page.getPixmap(clip=bbox)
            path = f"{TEMP_IMG_NAME}.{image_ext}"

            try:
                page_img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
                page_bytes = np.array(page_img).flatten()
                white_percent = len([b for b in page_bytes if b == 255]) / len(page_bytes)

                if white_percent > 0.5:
                    break

                if (page_img.getbbox()[3]+1) < current.getbbox()[3]:
                    #print(f'Unequal when checking {xref}')
                    continue

                current.save(open(path, "wb"))
                page_img.save(open(TEMP_PAGE_NAME, 'wb'))
                (val, loc) = open_cv_match(TEMP_PAGE_NAME, path)
                if max_match < val and loc == (0,0):
                    max_match = val
                    max_loc = loc
                #print(f'xref={xref}, max_match={max_match}, max_loc={max_loc}, w={w}, h={h}')
                if max_match >= 0.865:
                    break
            except:
                pass

        if max_match >= 0.865:
            reduced.append((xref, loc))
    reduced.sort(key=lambda x: x[1][1])
    return [r[0] for r in reduced]

def open_cv_match(page_path, img_path, debug=False):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    template = cv.imread(page_path, cv.IMREAD_GRAYSCALE)
    res = cv.matchTemplate(img, template, cv.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    if debug:
        from matplotlib import pyplot as plt
        w, h = template.shape[::-1]
        top_left = min_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(img,top_left, bottom_right, 0, 2)
        plt.subplot(121),plt.imshow(template)
        plt.title(''), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img)
        plt.title(''), plt.xticks([]), plt.yticks([])
        plt.show()
    return (1.0 - min_val, min_loc)

def get_blocks(page):
    text = page.getText('blocks')
    imgs = [img for img in page.getImageList(full=True) if img[3] > 20]
    imgs.sort(key=lambda x: page.getImageBbox(x)[1])
    return (text, imgs)

def get_page_data(page, id):
    name_bs = []
    i = 0
    text, imgs = get_blocks(page)
    while i < len(text):
        block = text[i]
        regex = '^' + str(id) + r'\n.+?' + date_re
        if re.match(regex, block[4], flags=re.S):
            name_bs.append(block[4])
            id+=1
        elif i+1 < len(text) and re.match('^' + str(id) + r'\n', block[4], flags=re.S):
            j = i+1
            name = block[4]
            while j < len(text) and not re.match(regex, name, flags=re.S):
                name += text[j][4]
                j+=1
            name_bs.append(name)
            i = j - 1
            id+=1
        else:
            pass
        i+=1
    if len(name_bs) != len(imgs):
        imgs = open_cv_reduce(page, imgs)
    else:
        imgs = [img[0] for img in imgs]

    assert len(name_bs) == len(imgs)
    return zip(name_bs, imgs)

if len(sys.argv) != 2:
    print(f'Usage: {sys.argv[0]} [PDF_FILE]', file=sys.stderr)
    sys.exit(1)

pdf_file = fitz.open(sys.argv[1])
if not os.path.isdir(IMAGES_FOLDER):
    os.mkdir(IMAGES_FOLDER)

id = 1
data = list()
for page_index in range(len(pdf_file)):
    page = pdf_file[page_index]
    text = page.getText('blocks')
    image_list = page.getImageList()
    contents = list(get_page_data(page, id))
    id += len(contents)

    for data_block, xref in contents:
        split = data_block.splitlines()
        number = split[0].strip()
        date = split[-1].strip()
        name = split[-2].strip()
        title = ' '.join(map(str.strip, split[1:-2]))

        base_image = pdf_file.extractImage(xref)
        image_bytes = base_image['image']
        image_ext = base_image['ext']
        img_path = f'{IMAGES_FOLDER}/{number}.{image_ext}'
        image = Image.open(io.BytesIO(image_bytes))
        image.save(open(img_path, "wb"))
        block = { 'id': number,
                  'name': name,
                  'title': title,
                  'date': date,
                  'img': img_path }
        data.append(block)

import json
import glob
with open('database.json', 'w') as d:
    d.write(json.dumps(data))

for tempfile in glob.glob('./temp*'):
    os.remove(tempfile)

print('Success!')
