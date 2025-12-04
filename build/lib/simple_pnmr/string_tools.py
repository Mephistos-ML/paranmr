'''
This submodule contains methods for manipulating strings
'''
import re
import xyz_py.atomic as atomic


def title(string: str) -> str:
    '''
    For a given string, appends ---- above and below

    Parameters
    ----------
    string: str
        string to add title lines to

    Returns
    -------
    str
        titled string
    '''

    titled = '\n'
    titled += '-' * (len(string) + 4)
    titled += '\n'
    titled += '- {} -\n'.format(string)
    titled += '-' * (len(string) + 4)
    titled += '\n'

    return titled


def subtitle(string: str) -> str:
    '''
    For a given string, appends ---- below

    Parameters
    ----------
    string: str
        string to add subtitle line beneath

    Returns
    -------
    str
        subtitled string
    '''

    subtitled = '\n{}\n'.format(string)
    subtitled += '-' * len(string)
    subtitled += '\n'

    return subtitled


def remove_numbers(string: str) -> str:
    """
    Removes numbers from a string

    Parameters
    ----------
    string: str
        String whose numbers will be removed

    Returns
    -------
    str
        Input string with numbers removed

    """

    no_digits = []
    for i in string:
        if not i.isdigit():
            no_digits.append(i)
        elif i.isdigit():
            continue
    result = ''.join(no_digits)

    return result


def remove_letters(string: str) -> str:
    """
    Removes letters from a string

    Parameters
    ----------
    string: str
        String whose letters will be removed

    Returns
    -------
    str
        Input string with letters removed

    """

    no_letters = []
    for i in string:
        if i.isdigit():
            no_letters.append(i)
        elif not i.isdigit():
            continue
    result = ''.join(no_letters)

    return result


def lab_adjust(label: str, shift: int) -> str:
    """
    Increments number in a label by shift
    label MUST be formatted ATOMNUMBER e.g. Ca33

    Parameters
    ----------
    label : str
        label to modify
    shift : int
        shift to apply to label number

    Returns
    -------
    str
        New label with shifted number
    """

    letters = remove_numbers(label)
    number = remove_letters(label)
    new_number = int(number) + shift

    new_lab = "{}{}".format(letters, new_number)

    return new_lab


def atom_range_to_list(atom_range: str) -> list[str]:
    '''
    Convert a string range of atom labels/numbers into a list of actual labels
    e.g. 'H1-H10' --> 'H1', 'H2', 'H3', 'H4', 'H5', ..., 'H10'

    Parameters
    ----------
    atom_range: str
        Atom range as string e.g. 'H1-H10'
    Returns
    -------
    list[str]
        Atom labels
    '''

    # Remove hyphen, get number and generate all labels in range
    start = int(re.search(r'\d+', atom_range.split('-')[0]).group(0))
    end = int(re.search(r'\d+', atom_range.split('-')[1]).group(0))
    ele = re.search(
        r'[a-zA-Z\s]+', atom_range.split('-')[1]
    ).group(0).capitalize()
    if ele not in atomic.elements:
        raise ValueError(f'Unknown nucleus type {ele}')
    else:
        atom_list = [
            f'{ele}{it}'
            for it in range(start, end + 1)
        ]
    return atom_list
