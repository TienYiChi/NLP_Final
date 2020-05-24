import os, csv

with open('resources/valid_partial.csv', 'r', newline='') as infile:
    data = list(csv.DictReader(infile, delimiter=';'))
    with open('resources/valid_textless.csv', 'w', newline='') as outfile:
        fieldnames = ['Index', 'Gold']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow({'Index': row['Index'], 'Gold': row['Gold']})