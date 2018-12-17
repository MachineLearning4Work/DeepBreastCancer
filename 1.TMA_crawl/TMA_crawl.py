'''
All rights reserved", Royan Institute for Stem Cell Biology and Technology,
Oct 2017  ( Mehdi Habibzadeh et al)
'''



import requests
import urllib
from os.path import expanduser, join
from bs4 import BeautifulSoup
import os

readSite =requests.get('https://tma.im/cgi-bin/viewStain.pl?stain_no=3325&view=Printable').text

soup = BeautifulSoup(readSite, 'lxml')


images=[]
tag = soup.findAll('img')
for x in tag:
    print(x.get('src'))


for x in tag[1:]:
    i = x['src']
    i = i.split('/')
    print(i)
    j='http://jpg.tma.im'+i[2]+'/'+i[3]
    d=os.path.join(i[2],i[3])
    m=os.path.join('http://jpg.tma.im',d)
    print(m)
    images.append(m)
print(images)

pathlink='CancerLinks.docx'
fullPathLink =os.path.join(expanduser('.'), pathlink)
outputImageLink= open(fullPathLink, "w")


for image in images:

    print("A given image URL : ")
    print(image)
    outputImageLink.write(image)
    outputImageLink.write('\n')
    resource = urllib.request.urlopen(image)

    try:
        path01 = resource.url.split("/")[-1]
        fullPath = join(expanduser('./ImgLink/'), path01)

        output = open(resource.url.split("/")[-1], "wb")
        output.write(resource.read())
    except Exception as e :
        print("Undefined Format",e)
    finally:
        output.close()




