# MasterThesis-RawWork

All of my data and work for the Masters Thesis in Applied Mathematics, in the field of Statistics and Neuroscience (Spring 2021).

Preliminary work and preperation was conducted as a course "Specialization project" (15 ETCS), see "Preliminary-work.pdf". The work was graded with an A (Top grade).

# Connecting to IDUN
(Must be on NTNU network or use VPN!)
https://www.hpc.ntnu.no/idun

Log on :

`ssh -l USERNAME idun-login1.hpc.ntnu.no`

The directory where you're allowed to do computations:

`cd /lustre1/work/USERNAME`

Here, clone into this github repo, by using 

`git clone https://github.com/amemil/MasterThesis-RawWork.git`

Connecting to IDUN when you are at home and not at NTNU:

1) `ssh <username>@markov.math.ntnu.no`
2) `ssh -l <username> idun-login1.hpc.ntnu.no`


## Creating a virtual environment in IDUN
https://www.hpc.ntnu.no/idun/getting-started-on-idun/modules

`$ type virtualenv`

`$ virtualenv datasci`

`$ source datasci/bin/activate`

`(datasci)$ pip install scipy numpy scikit-learn pandas matplotlib`

If you are already in the prosjektoppgave folder:

`source ../datasci/bin/activate`


## Creating a job in IDUN
https://www.hpc.ntnu.no/idun/getting-started-on-idun/running-jobs

Create a job.slurm file:

`vim job.slurm`
Use the key i to edit this file. Copy paste the file as it is on the help page. 
Account = ie-imf
Email: your ntnu email

When done, Ctrl+C, then write `:wq!` and hit enter.

`chmod u+x job.slurm`
`sbatch job.slurm`


# Making changes to the repository

First make sure you're up to date with the current repository:

`git pull`

Once you've made a change, use the command

`git add .`

(to add everything you've done) or 

`git add FILEYOUMADECHANGESTO.py`

`git commit -m "la til det her"`

`git push`



# Saving outputs from file runs

`python3 smth.py output > smth.txt`

# Move files 
Write in terminal: `mv <filename> <map>





# Downloading things (results) from Idun to your local computer

Emil:
`scp -r emilamy@idun-login1.hpc.ntnu.no:/lustre1/work/emilamy/prosjektoppgave/MAPPEN_DU_VIL_LASTE_NED STEDET_PÅ_PCEN_DER_DETTE_LAGRES`
