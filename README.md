# ramp-workflow
Toolkit for building analytics workflows on the top of pandas and scikit-learn. Primarily intended to feed RAMPs.

Workflow elements are file names. Most of them are python code files, they should have no extension. They will become editable on RAMP. Other files, e.g. external_data.csv or comments.txt whould have extensions. Editability fill be inferred from extension (e.g., txt is editable, csv is not, only uploadable). File names should contain no more than one '.'.