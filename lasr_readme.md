1. Environment
   If conda is installed, running the code below may install all the requirements.
   ```
   sh install.sh
   ```

2. Data Preparation
   ```
   sh preprocess.sh
   ```

3. Different settings

    #### Version1: Without GT camera
    Comment out Line194-Line205 in **dataloader/vidbase_cus.py**. And uncomment Line190-Line191    in    **dataloader/vidbase_cus.py**  

    #### Version2: With GT camera

    Comment out Line190-191 in **dataloader /vidbase_cus.py** and uncomment Line194-Line205 in **dataloader/vidbase_cus.py**

4. Running
    ```
    sbatch lasr.sbatch
    ```
    For each animal, the name should be changed in the **.sbatch** script.(3 places in total)   
\
    Remeber to change the index number and slurm setting parameters.
\
    The whole animal name list is contained in **val_split_list.py**



