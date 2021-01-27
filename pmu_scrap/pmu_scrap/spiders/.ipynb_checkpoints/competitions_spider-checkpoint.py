import scrapy
import os
import datetime
from scrapy.shell import inspect_response


class CompetitionsSpider(scrapy.Spider):
    name = "competitions"

    def start_requests(self):
        day = datetime.datetime(day=1, month=1, year=2014)
        while day < datetime.datetime.today():
        # while day < datetime.datetime(day=1, month=3, year=2014):
            print(day.date())
            if not os.path.isfile('data/competitions/{}.json'.format(day.strftime('%d%m%Y'))):
                yield scrapy.Request(
                    'https://online.turfinfo.api.pmu.fr/rest/client/1/programme/{}'.format(day.strftime('%d%m%Y')),
                    callback=self.save)

            day += datetime.timedelta(days=1)

    def save(self, response):
        day = response.request.url.split('/')[-1]
        # inspect_response(response,self)
        path = f'data/competitions/{day}.json'
        try:
            with open(path, 'w') as file:
                file.write(response.body.decode('utf-8'))
        except Exception as e:
            print(e)
            os.remove(path)
