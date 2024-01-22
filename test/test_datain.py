# test data munging code

from mavpdl1.data_examination.data_munging import PDL1Data


if __name__ == "__main__":
    data_p = "/Users/jculnan/va_data/pdl1_annotations-100_deidentified.csv"

    data = PDL1Data(data_p)
    print(data.data)