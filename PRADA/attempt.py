import json
import csv

# Opening JSON file
f = open('package.json', 'r', encoding='UTF-8')

# returns JSON object as a dictionary
data = json.load(f)

company = str(data).split("class=\"g-lockup\"")
length = len(company)

# create csv file
with open('angel.co/artificial-intelligence.csv', mode='w', encoding='utf-8') as csv_file:
    fieldnames = ['index', 'company', 'blurb', 'location', 'field', 'joined', 'followers']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(length):

        index = company[i].find(f'title="')
        name = company[i][index + 7:index + 50]
        name = name.split(sep="\"")[0]

        index = company[i].find("<div class=\"blurb\">\\n")
        blurb = company[i][index+21:index + 200]
        blurb = blurb.split(sep='\\n')[0]

        index = company[i].find(f'</a> &middot;')
        location = company[i][index-20:index]
        location = location.split(sep='>')[-1]

        index = company[i].find(f'</a>\\n</div>\\n</div>\\n</div>\\n</div>\\n<div class="column joined" ')
        field = company[i][index-50:index]
        field = field.split(sep='>')[-1]

        index = company[i].find(f'&rsquo')
        joined = company[i][index-4:index-1] + ' ' + company[i][index+7]

        index = company[i].find(f'Followers\\n')
        followers = company[i][index+40:index+45]
        followers = followers.split(sep='\\n')[0]

        writer.writerow({'index': i, 'company': name, 'blurb': blurb, 'location': location, 'field': field, 'joined': joined, 'followers': followers})

    print('finish')