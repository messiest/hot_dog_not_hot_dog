import os
import shutil
import urllib.request

import cv2


# credit to kmather73 from github for some of this code

# Go to image.net find your images then click download image urls
# Add that pages url to links

def store_raw_images(folders, links):
    pic_num = 1
    for link, folder in zip(links, folders):
        if not os.path.exists(folder):
            os.makedirs(folder)
        image_urls = str(urllib.request.urlopen(link).read())

        for i in image_urls.split('\\n'):
            try:
                urllib.request.urlretrieve(i, folder + "/" + str(pic_num) + ".jpg")
                img = cv2.imread(folder + "/" + str(pic_num) + ".jpg")

                # Do preprocessing if you want
                if img is not None:
                    cv2.imwrite(folder + "/" + str(pic_num) + ".jpg", img)
                    pic_num += 1

            except Exception as e:
                print(str(e))


def read_images_to_folder():
    links = [
        'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02472987',
        'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03405725',
        'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01316949',
        'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n10639238',
        'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n06255081',
        'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07865105',
        'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07690019',
        'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07697537'
    ]

    paths = ['humans', 'furniture', 'animals', 'sports', 'vehichle'
                                                         'chili_dog', 'frankfurter', 'hot_dog']

    os.mkdir('not_hotdog')
    os.mkdir('hotdog')

    store_raw_images(paths, links)

    for path in paths[:4]:
        files = os.listdir(path)

        for f in files:
            shutil.move(path + f, './not_hotdog')

    for path in paths[5:]:
        files = os.listdir(path)

        for f in files:
            shutil.move(path + f, './hotdog')
