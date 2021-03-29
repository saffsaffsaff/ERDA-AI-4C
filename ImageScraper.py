import requests
from bs4 import BeautifulSoup
import os

# = 'https://www.jetphotos.com/showphotos.php?aircraft=all&airline=all&country-location=location%253BAmsterdam%2520Schiphol%2520Airport%2520-%2520EHAM&photographer-group=all&category=all&keywords-type=all&keywords-contain=3&keywords=&photo-year=all&genre=all&search-type=Advanced&sort-order=0'

def aeroplanedatabase(url, folder):
    try:
        os.mkdir(os.path.join(os.getcwd(), folder))
    except:
        pass
    os.chdir((os.path.join(os.getcwd(), folder)))
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')

    images = soup.find_all('img')

#loop through all the pictures/ img on webpage
    for image in images[4:]:
        #name = alt text
        name = image['alt']
        #image link = source link from css
        link = image['src']
        #make link usable
        link1 = link.replace('//', 'http://', 1)
        #make name usable
        print(name.split('----'))
        #save images as jpeg in assigned folder
        with open(name.replace('/', '') + '.JPG', 'wb') as f:
            im = requests.get(link1)
            f.write(im.content)



aeroplanedatabase('https://www.jetphotos.com/showphotos.php?aircraft=all&airline=all&country-location=location%253BAmsterdam%2520Schiphol%2520Airport%2520-%2520EHAM&photographer-group=all&category=all&keywords-type=all&keywords-contain=3&keywords=&photo-year=all&genre=all&search-type=Advanced&sort-order=0', 'page 1')
#aeroplanedatabase('https://www.jetphotos.com/showphotos.php?aircraft=all&airline=all&country-location=location%253BAmsterdam%2520Schiphol%2520Airport%2520-%2520EHAM&photographer-group=all&category=all&keywords-type=all&keywords-contain=3&keywords=&photo-year=all&genre=all&search-type=Advanced&sort-order=0&page=2', 'page2.1')