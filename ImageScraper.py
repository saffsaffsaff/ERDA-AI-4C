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
    callsign = []
    manufacturer = []
    type = []
    carrier = []

    for image in images[4:]:
        name = image['alt']
        link = image['src']
        link1 = link.replace('//', 'http://', 1)
        print(name.split('----'))
        callsign.append(name.split('----'))


        #with open(name.replace('/', '') + '.JPG', 'wb') as f:
        #    im = requests.get(link1)
        #   f.write(im.content)



aeroplanedatabase('https://www.jetphotos.com/showphotos.php?aircraft=all&airline=all&country-location=location%253BAmsterdam%2520Schiphol%2520Airport%2520-%2520EHAM&photographer-group=all&category=all&keywords-type=all&keywords-contain=3&keywords=&photo-year=all&genre=all&search-type=Advanced&sort-order=0', 'page 1')
#aeroplanedatabase('https://www.jetphotos.com/showphotos.php?aircraft=all&airline=all&country-location=location%253BAmsterdam%2520Schiphol%2520Airport%2520-%2520EHAM&photographer-group=all&category=all&keywords-type=all&keywords-contain=3&keywords=&photo-year=all&genre=all&search-type=Advanced&sort-order=0&page=2', 'page2.1')