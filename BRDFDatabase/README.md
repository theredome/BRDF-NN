<!--
Created by Mitsubishi Electric Research Laboratories (MERL), 2003, 2007, 2023

SPDX-License-Identifier: CC-BY-SA-4.0
-->

# BRDF Database

## Introduction
The MERL BRDF database contains reflectance functions of 100 different materials. Each reflectance function is stored as a densely measured Bidirectional Reflectance Distribution Function (BRDF).

Sample code to read the data is included with the database (`readBRDF.cpp`). Note that parameterization of theta-half has changed.

## Dataset Organization

The size of the unzipped dataset is ~3.25GB. Folder `brdfs` contains the data. Folder `code` contains sample code.

## Citation

If you use the BRDF dataset in your research, please cite our contribution:

```BibTex
@article {Matusik:2003,
	author = "Wojciech Matusik and Hanspeter Pfister and Matt Brand and Leonard McMillan",
	title = "A Data-Driven Reflectance Model",
	journal = "ACM Transactions on Graphics",
	year = "2003",
	month = jul,
	volume = "22",
	number = "3",
	pages = "759-769"
}
```

## License

The BRDF dataset is released under [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

All data:

```
Created by Mitsubishi Electric Research Laboratories (MERL), 2003, 2007, 2023

SPDX-License-Identifier: CC-BY-SA-4.0
```
