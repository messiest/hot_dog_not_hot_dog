import os
import urllib.request
import cv2

# credit to kmather73 from github for some of this code

# Go to image.net find your images then click download image urls
# Add that pages url to links

def get_images(folders, links):
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


def main():
    links = ['http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01846331']
    paths = ['ducks']

    get_images(paths, links)


if __name__ == '__main__':
    main()
