# Homework, 30-Oct-2026

In the previous weeks your managed to translate *text* and *pdf* files.

This week you will put everything together, to create a useful command-line translation tool.

- Create a Python program named ```eng2greek.py```
- Use the ```argparse``` library.

When you run the program in the terminal with the -h option ```python eng2greek.py -h``` you should get the following message:

```commandline
usage: eng2greek.py [-h] [-o OUTFILE] [-w] infile

Translate a txt or pdf file from English to Greek.

positional arguments:
  infile                the input English file; can be .txt or .pdf

options:
  -h, --help            show this help message and exit
  -o OUTFILE, --outfile OUTFILE
                        the output Greek file (default: output.txt)
  -w, --write           overwrite the output file, even if file exists.
```
### Input
The program needs an input file ```infile```. 
- If ```infile``` is a .txt file, then execute your previous code for text files. 
- If ```infile``` is a .pdf file, then execute your previous code for pdf files. 
- Else, print an error message and exit.

### Output
- By default, the translation should be written to ```output.txt```
- However, the user may optionally choose a different ```outfile``` by using the -o argument. For example ```-o greek.txt``` will write the translation to ```greek.txt```.

### File overwrite
- By default, if ```outfile``` already exists, the program should print an error message and exit, without overwriting the existing file.
- However, if the user sets the ```-w``` flag, the program is allowed to overwrite ```outfile```.

### Example usage
```python eng2greek.py document.pdf -o miceman.txt -w```
- Open English file ```document.pdf```
- Write the translation to file ```miceman.txt```
- (because of ```-w```): If ```miceman.txt```, overwrite it.

```python eng2greek.py image.jpg```
- Should print an error message: "Sorry, I can only open .txt and .pdf files."

```python eng2greek.py input.txt```
- Open English file ```input.txt```
- If ```output.txt``` already exists, then print error message "File output.txt already exists", and exit the program.
- Else, write the translation to file ```output.txt```




## AI use policy
- You are allowed to ask AI to help you with specific questions, for example *"How can I read a text file in Python?"* or *"How can I split a long string into paragraphs?"*
- You are not allowed to ask AI to write the entire program.
- You must be able to **explain** every line of you