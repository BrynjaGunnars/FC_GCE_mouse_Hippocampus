#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 14:31:47 2020

@author: brynjagunnarsdottir
"""
#Functions and script used to get correlation maps from AGEA, note: time consuming and
#there is probably a much better way to get the maps

import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import ActionChains
import re
import requests
import webbrowser


#Function to find if atlas cursor is on the right structure

def findWholeWord(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

#Function with same purpose as findWholeWord, should be easier to run for computer than findWholeWord
#but wasn't in practice

def finnaOrd(a,b):
    r = requests.get(a)
    r = r.text
    return re.search(b, r)

#Go to website, not necessary but makes it possible to close about pop-up button before entering the loop
#Need webdriver, here webdriver for chrome is used   
driver = webdriver.Chrome('path/to/driver')  
driver.get('https://mouse.brain-map.org/agea');

#Close pop-up

closeAbout = driver.find_element_by_id("closeAboutButton")
ActionChains(driver).move_to_element(closeAbout).click(closeAbout).perform()


#Loop, using findWholeWord function. Loop through x, y and z coordinates of interest
 
x=8800    
while x <= 8800:
    y = 3200
    while y <= 6400:
        z= 3600
        while z <= 5800:
            stringX = str(x)
            stringY = str(y)
            stringZ = str(z)
            locLink = 'https://mouse.brain-map.org/agea?seed=P56,' + stringX + ',' + stringY + ',' + stringZ + '&map1=P56,' + stringX + ',' + stringY + ',' + stringZ + ',[0.7,1]'
            driver.get(locLink);
            #Find where cursor is located structure-wise
            label = driver.find_element_by_xpath("/html/body/div[@class='siteContent']/table[@class='pageContent']/tbody/tr/td/div[@id='mainContent']/div[@id='panelsContainer']/div[@id='refAtlasPanel']/div[2]/div[@class='imageControlsContainer']/div[@class='imageControlsInnerContainer'][1]/div[@class='annotationLabel']/div[@class='annotationNameBlock']")
            texti = label.text
            print(texti)
            if findWholeWord('Dentate')(texti):
                dloadLink = 'http://mouse.brain-map.org/agea/data/P56/download?seed='+ stringX + ','+stringY + ','+ stringZ +'&age=P56'
                webbrowser.open(dloadLink)
            elif findWholeWord('CA1')(texti):
                dloadLink = 'http://mouse.brain-map.org/agea/data/P56/download?seed='+ stringX + ','+stringY + ','+ stringZ +'&age=P56'
                webbrowser.open(dloadLink)
            elif findWholeWord('CA2')(texti):
                dloadLink = 'http://mouse.brain-map.org/agea/data/P56/download?seed='+ stringX + ','+stringY + ','+ stringZ +'&age=P56'
                webbrowser.open(dloadLink)
            elif findWholeWord('CA3')(texti):
                dloadLink = 'http://mouse.brain-map.org/agea/data/P56/download?seed='+ stringX + ','+stringY + ','+ stringZ +'&age=P56'
                webbrowser.open(dloadLink)
            z += 200
        y += 200
    x += 200
    

#Loop-de-loop, using finnaOrd function, loop through x, y and z coordinates of interest
    
x=7800    
while x <= 7800:
    y = 1400
    while y <= 6400:
        z= 8800
        while z <= 10000:
            stringX = str(x)
            stringY = str(y)
            stringZ = str(z)
            locLink = 'https://mouse.brain-map.org/agea/data/P56/voxel_info?seed='+ stringX + ',' + stringY + ',' + stringZ 
            #Find where cursor is located structure-wise
            if finnaOrd(locLink, 'Dentate'):
                dloadLink = 'http://mouse.brain-map.org/agea/data/P56/download?seed='+ stringX + ','+stringY + ','+ stringZ +'&age=P56'
                webbrowser.open(dloadLink)
            elif finnaOrd(locLink,'CA1'):
                dloadLink = 'http://mouse.brain-map.org/agea/data/P56/download?seed='+ stringX + ','+stringY + ','+ stringZ +'&age=P56'
                webbrowser.open(dloadLink)
            elif finnaOrd(locLink,'CA2'):
                dloadLink = 'http://mouse.brain-map.org/agea/data/P56/download?seed='+ stringX + ','+stringY + ','+ stringZ +'&age=P56'
                webbrowser.open(dloadLink)
            elif finnaOrd(locLink,'CA3'):
                dloadLink = 'http://mouse.brain-map.org/agea/data/P56/download?seed='+ stringX + ','+stringY + ','+ stringZ +'&age=P56'
                webbrowser.open(dloadLink)
            print(stringX + ', ' + stringY + ', '+ stringZ)
            z += 200
        y += 200
    x += 200
   
    

