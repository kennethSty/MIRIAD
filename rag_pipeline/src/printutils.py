from termcolor import colored


def pcprint(config):
    for key, value in config.items():
        print(colored(f"{key}:", "blue", attrs=["bold"]))
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                print(colored(f"  {sub_key}:", "green", attrs=["bold"]), end=" ")
                print(colored(sub_value, "yellow"))
        else:
            print(colored(f"  {value}", "yellow"))


def separator():
    print(colored("\n" + "=" * 25 + "\n", "red"))

def print_comparison(list_1, list_2, span=110):
    # print the comparison of two lists
    # the length of the two lists could be different
    # the comparison will be printed line by line
    # the span is the length of the line
    # the two lists will be printed side by side
    # make sure to take care of the case when the length of the two lists are different
    # if the length of the two lists are different, the shorter list will be padded with empty strings
    
    # get the length of the two lists
    len_1 = len(list_1)
    len_2 = len(list_2)
    
    # get the maximum length of the two lists
    max_len = max(len_1, len_2)
    
    # pad the two lists with empty strings
    list_1 += [''] * (max_len - len_1)
    list_2 += [''] * (max_len - len_2)
    
    # print the comparison line by line
    for i in range(max_len):
        print(list_1[i].ljust(span) + list_2[i])
    print(i)