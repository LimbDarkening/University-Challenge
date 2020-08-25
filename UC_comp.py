import wikipedia as wiki
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

        
def get_tables(title):
    """Get result tables from wikipedia titles"""
    page = wiki.page(title)
    soup = BeautifulSoup(page.html(), 'html.parser')
    tables = soup.findAll("table", class_ = "wikitable")
    
    problem_years = {"University Challenge 2015–16",
                     "University Challenge 2016–17",
                     "University Challenge 2017–18",
                     "University Challenge 2018–19",
                     "University Challenge 2019–20"
                     }
    if title in problem_years:
        return tables[:6]
    #edge case
    if title == "University Challenge 2009–10":
        return tables[1:]
    
    else:
        return tables
    
exceptions = {"University of St Andrews": "University of St Andrews",
                  "City University": "City, University of London"
                  }
def get_name(title):
            
    if title in exceptions:
        title = exceptions.get(title)
        return title
    
    page = wiki.page(title)
    exceptions[title] = page.title
    return page.title

#Compile list of years
page = wiki.page(title = "University Challenge 1994–95")      
links = page.links[29:54]
links = ["University Challenge 1994–95"] + links

years = [get_tables(title) for title in links]

#scraping
Team1 = []
Team1_score = []
Team2_score = []
Team2 = []
Date = []

for year in years:
    for table in year:
        for row in table.findAll('tr')[1:]:
            cells=row.findAll('td')
            Team1.append(cells[0].find(text=True))
            Team1_score.append(cells[1].find(text=True))
            Team2_score.append(cells[2].find(text=True))
            Team2.append(cells[3].find(text=True))
            if len(cells)==5:
                Date.append(cells[4].find(text=True))
            else:
                Date.append(cells[5].find(text=True))
                
Team1[29] = 'University of Lancaster'  

#Rounds
begining = ['1R']*12 + ['2R']*8 + ['QF']*4 + ['SF']*2 + ['F']
mid = ['1R']*14 + ['HSLP']*2 + ['2R']*8 + ['QF']*4 + ['SF']*2 + ['F']
end = ['1R']*14 + ['HSLP']*2 + ['2R']*8 + ['QF']*10 + ['SF']*2 + ['F']

early_years = begining * 4
mid_years = mid * 11
late_years = end * 11
rounds = early_years + mid_years + late_years
'''
#Clean
unique_names = set(Team1 + Team2)
print(len(unique_names))
#reduc_names = [get_name(name) for name in unique_names]
reduc_names = []
for name in unique_names:
    reduc_names.append(get_name(name))
    
print(len(reduc_names))
'''
#Compile                
Data = pd.DataFrame([get_name(name) for name in Team1], columns = ['Team1'])
Data['Team1_score'] = [score.rstrip("\n") for score in Team1_score]                
Data['Team2_score'] = [score.rstrip("\n") for score in Team2_score]           
Data['Team2'] = [get_name(name) for name in Team2]
Data['Date'] = [datetime.strptime(day.rstrip("\n"),"%d %B %Y") for day in Date]       
Data['Round'] = rounds

#Data.to_csv('UC_database.txt')                
                    
            