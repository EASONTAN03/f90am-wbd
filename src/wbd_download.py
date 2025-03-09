import wbdata
import pandas as pd
import datetime

# Function to download data from World Bank API
def download_world_bank_data(indicators, country_codes, start_year, end_year, output_file):
    """
    Downloads tabular and time-series data from the World Bank API.

    Parameters:
        indicators (dict): A dictionary of indicators {"indicator_code": "indicator_name"}.
        country_codes (list): List of country codes to include in the data.
        start_year (int): Start year for the data.
        end_year (int): End year for the data.
        output_file (str): Path to save the downloaded data as a CSV file.

    Returns:
        None
    """
    # Set the date range
    data_date_range = (datetime.datetime(start_year, 1, 1), datetime.datetime(end_year, 12, 31))

    # Fetch data
    print("Fetching data from the World Bank API...")
    data = wbdata.get_dataframe(indicators, country=country_codes, date=data_date_range, parse_dates=True)

    # Reset index and save to CSV
    data.reset_index(inplace=True)
    data.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

# Define indicators of interest (you can look up indicators in the metadat file)
# Each idicator has an ID, and you can assign to it a label that will be used as column name in the dataFrame

indicators = {
    "NY.GDP.PCAP.PP.KD": "GDPpc_2017$", #"GDP per capita, PPP (constant 2017 US$)
    "SP.POP.TOTL": "Population_total",
    "SP.DYN.LE00.IN": "Life_exectancy"
    ,"SE.ADT.LITR.ZS": "Literacy_rate"
    ,"SL.UEM.TOTL.ZS": "Unemploymlent_rate"
    ,"EG.USE.PCAP.KG.OE": "Access_electricity"
    ,"SP.DYN.TFRT.IN": "Fertility_rate"
    ,"SI.POV.NAHC": "Poverty_ratio"
    ,"SE.PRM.ENRR": "Primary_school_enrolmet_rate"
    #,"NE.GDI.TOTL.ZS": "Gross_capital_formation"
    #,"IT.NET.USER.ZS": "Internet_users"
    #,"SE.XPD.TOTL.GD.ZS": "Gov_education_investment"
    #,"EN.ATM.CO2E.PC": "CO2_emissions"
    #,"SI.POV.GINI": "Income_inequality"
    #,"SH.H2O.BASW.ZS": "Access_to_water"
    ,"EG.USE.PCAP.KG.OE": "Energy_use"
    #,"NE.EXP.GNFS.CD": "Exports_c$", #Exports of goods and services (current US$)
    ,"NE.EXP.GNFS.KD": "Exports_2017$" #Exports of goods and services (current US$)
    #,"SE.XPD.PRIM.PC.ZS":  "Expenditure_primary" #Government expenditure per student, primary (% of GDP per capita)
    #,"SE.XPD.SECO.PC.ZS":  "Expenditure_secodary" #Government expenditure per student, secondary (% of GDP per capita)
    #,"SE.XPD.TERT.PC.ZS":  "Expenditure_tertiary" #"Government expenditure per student, tertiary (% of GDP per capita)"
    }



expression='^(?!(?:World|Upper middle income|Sub\-Saharan Africa excluding South Africa|Sub\-Saharan Africa|Latin America \& Caribbean|IDA countries in Sub\-Saharan Africa classified as fragile situations|IDA\scountries\sin\sSub\-Saharan\sAfrica\snot\sclassified\sas\sfragile\ssituations|Sub\-Saharan Africa excluding South Africa and Nigeria|South Asia \(IDA \& IBRD\)|Sub-Saharan Africa \(IDA \& IBRD countries\)|Middle East \& North Africa \(IDA \& IBRD countries\)|Latin America \& the Caribbean \(IDA \& IBRD countries\)|Europe \& Central Asia \(IDA \& IBRD countries\)|East Asia \& Pacific \(IDA \& IBRD countries\)|Sub\-Saharan Africa|Sub\-Saharan Africa \(excluding high income\)|Resource rich Sub\-Saharan Africa countries|IDA countries not classified as fragile situations\, excluding Sub-Saharan Africa|Non\-resource rich Sub\-Saharan Africa countries|Middle East \& North Africa \(excluding high income\)|Middle East \(developing only\)|Latin America \& Caribbean \(excluding high income\)|IBRD\, including blend|Heavily indebted poor countries \(HIPC\)|IDA countries classified as fragile situations\, excluding Sub\-Saharan Africa|Europe \& Central Asia \(excluding high income\)|East Asia \& Pacific \(excluding high income\)|Sub\-Saharan Africa \(IDA\-eligible countries\)|IDA countries in Sub\-Saharan Africa classified as fragile situations|South Asia \(IDA\-eligible countries\)|IDA countries in Sub\-Saharan Africa not classified as fragile situations|Middle East \& North Africa \(IDA\-eligible countries\)|Latin America \& the Caribbean \(IDA\-eligible countries\)|Europe \& Central Asia \(IDA\-eligible countries\)|East Asia \& Pacific \(IDA\-eligible countries\)|South Asia \(IFC classification\)|Middle East and North Africa \(IFC classification\)|Latin America and the Caribbean \(IFC classification\)|Europe and Central Asia \(IFC classification\)|East Asia and the Pacific \(IFC classification\)|Sub\-Saharan Africa \(IFC classification\)|Sub\-Saharan Africa \(IBRD\-only countries\)|Latin America \& the Caribbean \(IBRD\-only countries\)|Middle East \& North Africa \(IBRD\-only countries\)|East Asia \& Pacific \(IBRD\-only countries\)|IBRD countries classified as high income|Europe \& Central Asia \(IBRD\-only countries\)|Sub\-Saharan Africa \(IDA \& IBRD countries\)|Sub-Saharan Africa \(excluding high income\)|Sub\-Saharan Africa|South Asia \(IDA \& IBRD\)|South Asia|Small states|Pre\-demographic dividend|Post\-demographic dividend|Pacific island small states|Other small states|OECD members|Not classified|North America|Middle income|Middle East \& North Africa \(IDA \& IBRD countries\)|Middle East \& North Africa \(excluding high income\)|Middle East \& North Africa|Lower middle income|Low income|Low \& middle income|Least developed countries\: UN classification|Latin America \& the Caribbean \(IDA \& IBRD countries\)|Latin America \& Caribbean \(excluding high income\)|Latin America \& Caribbean|Late\-demographic dividend|IDA total|IDA only|IDA blend|IDA \& IBRD total|IBRD only|High income|Heavily indebted poor countries \(HIPC\)|Fragile and conflict affected situations|European Union|Europe \& Central Asia \(IDA \& IBRD countries\)|Europe \& Central Asia \(excluding high income\)|Europe \& Central Asia|Euro area|East Asia \& Pacific \(IDA \& IBRD countries\)|East Asia \& Pacific \(excluding high income\)|East Asia \& Pacific|Early\-demographic dividend|Central Europe and the Baltics|Caribbean small states|Arab World|Africa Western and Central|Africa Eastern and Southern)$).*$'
data = wbdata.get_countries(query=expression)
country_codes=[]
for item in data:
    if item["id"]!='DNS' and item["id"]!='DSF':
        country_codes.append(item["id"])

#country_codes = ["USA", "CHN", "IND", "BRA", "RUS"]  # Example countries
start_year = 1980
end_year = 2023
output_file = "world_bank_data_dev.csv"

# Download and save data
download_world_bank_data(indicators, country_codes, start_year, end_year, output_file)
