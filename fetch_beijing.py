"""Download the UCI Beijing Multi-Site Air Quality data set (#501) into bj_data/ (run from the repository root).
Needs internet access. UNTESTED from the authoring sandbox (blocked); if the URL has changed, download the zip manually from
https://archive.ics.uci.edu/dataset/501 and put the 12 PRSA_Data_*.csv files in bj_data/."""
import urllib.request, zipfile, io, os, glob, shutil
URL='https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip'
os.makedirs('bj_data',exist_ok=True)
data=urllib.request.urlopen(URL,timeout=300).read()
z=zipfile.ZipFile(io.BytesIO(data))
z.extractall('bj_tmp')
for inner in glob.glob('bj_tmp/**/*.zip',recursive=True): zipfile.ZipFile(inner).extractall('bj_tmp/inner')
files=glob.glob('bj_tmp/**/PRSA_Data_*.csv',recursive=True)
for f in files: shutil.copy(f,'bj_data/')
shutil.rmtree('bj_tmp'); print(len(files),'files in bj_data/ (expected 12)')
