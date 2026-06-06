df = pd.read_csv()

df['Departure Time'] = df['departure_time'].strtime('%H:%M')