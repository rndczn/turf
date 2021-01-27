import scrapy
import os
import datetime
import json
from scrapy.shell import inspect_response

class ParticipantsSpider(scrapy.Spider):
    name = "participants"

    def start_requests(self):
        for file_name in os.listdir('data/competitions'):
            if not file_name.endswith('.json'):
                continue
            with open('data/competitions/'+file_name,'r') as file:
                data = json.load(file)
                day = file_name.replace('.json','')
                for r, reunion in enumerate(data['programme']['reunions']):
                    for c, course in enumerate(reunion['courses']):
                        if not os.path.isfile(f'data/participants/{day}_R{course["numReunion"]}_C{d}.json'):
                            yield scrapy.Request(
                                f'https://online.turfinfo.api.pmu.fr/rest/client/1/programme/{day}/R{course["numReunion"]}/C{course["numOrdre"]}/participants',
                                callback=self.save)

    def save(self, response):
        url = response.request.url.split('/')
        day = url[-4]
        r = url[-3]
        c = url[-2]
        # inspect_response(response,self)
        path = f'data/participants/{day}_{r}_{c}.json'
        try:
            with open(path, 'w') as file:
                file.write(response.body.decode('utf-8'))
        except Exception as e:
            print(e)
            os.remove(path)
