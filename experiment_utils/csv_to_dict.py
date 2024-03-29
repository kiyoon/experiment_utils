import csv

def csv_to_dict(csv_path, type_convert = None):
    csv_dict = {}
    fieldnames = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for fieldname in reader.fieldnames:
            csv_dict[fieldname] = []
            fieldnames.append(fieldname)

        for row in reader:
            for key in row.keys():
                if row[key]:
                    if type_convert:
                        if type(type_convert) == dict:
                            csv_dict[key].append(type_convert[key](row[key]))
                        elif type(type_convert) == type:
                            csv_dict[key].append(type_convert(row[key]))
                        else:
                            raise ValueError('Wrong type_convert value: expected None, type, or dict of fieldname and type pairs but got %s.' % str(type_convert))
                    else:   # None: just string
                        csv_dict[key].append(row[key])
                else:
                    # if row[key] is empty
                    csv_dict[key].append(None)

    return csv_dict, fieldnames

