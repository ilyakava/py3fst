#!/usr/bin/env python
# coding=utf-8

"""https://github.com/pettarin/ipapy

This file can be used to convert the phones used in TIMIT
to the IPA phonetic notation.

Links:
IPA unicode: https://www.phon.ucl.ac.uk/home/wells/ipa-unicode.htm
unicode ipa phones with sounds: http://www.antimoon.com/how/pronunc-soundsipa.htm
dat source: https://raw.githubusercontent.com/pettarin/ipapy/master/ipapy/data/arpabet.dat
phone comparisons table: https://www.isip.piconepress.com/projects/switchboard/doc/education/phone_comparisons/index.html
text of words to ipa phones text: https://tophonetics.com/
TIMIT phone description: https://catalog.ldc.upenn.edu/docs/LDC93S1/PHONCODE.TXT
"""


import os
import io
import sys

PY2 = (sys.version_info[0] == 2)

DATA_FILE_CODEPOINT_JOINER = u"_"
"""
Character to specify Unicode compound strings,
e.g. 0070_032A or U+0070_U+032A = LATIN SMALL LETTER P + COMBINING BRIDGE BELOW
"""

DATA_FILE_CODEPOINT_SEPARATOR = u" "
"""
Separator between Unicode codepoints or
Unicode compound strings for a given IPAChar
"""

DATA_FILE_COMMENT = u"#"
""" Ignore lines starting with this character """

DATA_FILE_FIELD_SEPARATOR = u","
""" Field separator for the data file """

DATA_FILE_VALUE_NOT_AVAILABLE = u"N/A"
""" Placeholder for an IPAChar not encoded in Unicode """

DATA_FILE_ASCII_NUMERICAL_CODEPOINT_START = u"00"
""" Numerical codepoints in ASCII fields must start with this string """

DATA_FILE_ASCII_UNICODE_CODEPOINT_START = u"U+"
""" Unicode codepoints in ASCII fields must start with this string """

def int_to_unichr(codepoint):
    """
    Return the Unicode character with the given codepoint,
    given as an integer.
    Example::
        97 => a
    :param int codepoint: the Unicode codepoint of the desired character
    :rtype: (Unicode) str
    """
    if PY2:
        return unichr(codepoint)
    return chr(codepoint)

def hex_to_unichr(hex_string):
    """
    Return the Unicode character with the given codepoint,
    given as an hexadecimal string.
    Return ``None`` if ``hex_string`` is ``None`` or is empty.
    Example::
        "0061"   => a
        "U+0061" => a
    :param str hex_string: the Unicode codepoint of the desired character
    :rtype: (Unicode) str
    """
    if (hex_string is None) or (len(hex_string) < 1):
        return None
    if hex_string.startswith("U+"):
        hex_string = hex_string[2:]
    return int_to_unichr(int(hex_string, base=16))

def convert_unicode_field(string):
    """
    Convert a Unicode field into the corresponding list of Unicode strings.
    The (input) Unicode field is a Unicode string containing
    one or more Unicode codepoints (``xxxx`` or ``U+xxxx`` or ``xxxx_yyyy``),
    separated by a space.
    :param str string: the (input) Unicode field
    :rtype: list of Unicode strings
    """
    values = []
    for codepoint in [s for s in string.split(DATA_FILE_CODEPOINT_SEPARATOR) if (s != DATA_FILE_VALUE_NOT_AVAILABLE) and (len(s) > 0)]:
        values.append(u"".join([hex_to_unichr(c) for c in codepoint.split(DATA_FILE_CODEPOINT_JOINER)]))
    return values

def convert_ascii_field(string):
    """
    Convert an ASCII field into the corresponding list of Unicode strings.
    The (input) ASCII field is a Unicode string containing
    one or more ASCII codepoints (``00xx`` or ``U+00xx`` or
    an ASCII string not starting with ``00`` or ``U+``),
    separated by a space.
    :param str string: the (input) ASCII field
    :rtype: list of Unicode strings
    """
    values = []
    for codepoint in [s for s in string.split(DATA_FILE_CODEPOINT_SEPARATOR) if (s != DATA_FILE_VALUE_NOT_AVAILABLE) and (len(s) > 0)]:
        #if DATA_FILE_CODEPOINT_JOINER in codepoint:
        #    values.append(u"".join([hex_to_unichr(c) for c in codepoint.split(DATA_FILE_CODEPOINT_JOINER)]))
        if (codepoint.startswith(DATA_FILE_ASCII_NUMERICAL_CODEPOINT_START)) or (codepoint.startswith(DATA_FILE_ASCII_UNICODE_CODEPOINT_START)):
            values.append(hex_to_unichr(codepoint))
        else:
            values.append(codepoint)
    return values

def convert_raw_tuple(value_tuple, format_string):
    """
    Convert a tuple of raw values, according to the given line format.
    :param tuple value_tuple: the tuple of raw values
    :param str format_string: the format of the tuple
    :rtype: list of tuples
    """
    values = []
    for v, c in zip(value_tuple, format_string):
        if v is None:
            # append None
            values.append(v)
        elif c == u"s":
            # string
            values.append(v)
        elif c == u"S":
            # string, split using space as delimiter
            values.append([s for s in v.split(u" ") if len(s) > 0])
        elif c == u"i":
            # int
            values.append(int(v))
        elif c == u"U":
            # Unicode
            values.append(convert_unicode_field(v))
        elif c == u"A":
            # ASCII
            values.append(convert_ascii_field(v))
        #elif c == u"x":
        #    # ignore
        #    pass
    return tuple(values)

def load_data_file(
    file_path,
    file_path_is_relative=False,
    comment_string=u"#",
    field_separator=u",",
    line_format=None
):
    """
    Load a data file, with one record per line and
    fields separated by ``field_separator``,
    returning a list of tuples.
    It ignores lines starting with ``comment_string`` or empty lines.
    If ``values_per_line`` is not ``None``,
    check that each line (tuple)
    has the prescribed number of values.
    :param str file_path: path of the data file to load
    :param bool file_path_is_relative: if ``True``, ``file_path`` is relative to this source code file
    :param str comment_string: ignore lines starting with this string
    :param str field_separator: fields are separated by this string
    :param str line_format: if not ``None``, parses each line according to the given format
                            (``s`` = string, ``S`` = split string using spaces,
                            ``i`` = int, ``x`` = ignore, ``U`` = Unicode, ``A`` = ASCII)
    :rtype: list of tuples
    """
    raw_tuples = []
    if file_path_is_relative:
        file_path = os.path.join(os.path.dirname(__file__), file_path)
    with io.open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if (len(line) > 0) and (not line.startswith(comment_string)):
                raw_list = line.split(field_separator)
                if len(raw_list) != len(line_format):
                    raise ValueError("Data file '%s' contains a bad line: '%s'" % (file_path, line))
                raw_tuples.append(tuple(raw_list))
    if (line_format is None) or (len(line_format) < 1):
        return raw_tuples
    return [convert_raw_tuple(t, line_format) for t in raw_tuples]

def get_phn2ipa_dict():
    phn2ipa_dict = {}
    ipa_arpanet = load_data_file(file_path="arpabet.dat", file_path_is_relative=True, line_format=u"UA")
    for ipa_, arpanet_ in ipa_arpanet:
        phn2ipa_dict[arpanet_[0].lower()] = ipa_[0]
    # extras
    phn2ipa_dict['h#'] = '/'
    phn2ipa_dict['ax-h'] = '/'
    phn2ipa_dict['kcl'] = '/'
    phn2ipa_dict['dcl'] = '/'
    phn2ipa_dict['bcl'] = '/'
    phn2ipa_dict['epi'] = '/'
    phn2ipa_dict['gcl'] = '/'
    phn2ipa_dict['hv'] = '/'
    phn2ipa_dict['pau'] = '/'
    phn2ipa_dict['pcl'] = '/'
    phn2ipa_dict['tcl'] = '/'
    return phn2ipa_dict

def remove_consecutive_repeats(inp_str, sep='-'):
    out = ['h#']
    for el in inp_str.split('-'):
        if out[-1] == el:
            continue
        else:
            out.append(el)
    return out

def phn_str_2_ipa_str(inp_str, sep='-'):
    phns = remove_consecutive_repeats(inp_str, sep='-')
    output = []
    for c in phns:
        output.append(phn2ipa[c])
    return output

if __name__ == '__main__':
    true = phn_str_2_ipa_str('ay-ay-ay-ay-ay-z-z-z-z-z-ix-ng-ng-ng-ng-s-s-s-s-tcl-tcl-t-t-r-r-iy-iy-iy-m-m-z-z-z-z-ax-v-v-dcl-dcl-dcl-dcl-ay-ay-ay-ay-ay-ay-ay-m-m-m-z-z-z-z-z-ng-ng-ng-kcl-kcl-kcl')
    print(true)
    pred = phn_str_2_ipa_str('aa-ay-aa-ah-ah-s-z-s-z-z-ix-n-ix-n-z-z-s-sh-s-tcl-f-sh-q-ay-nx-n-eh-ix-n-n-z-s-z-z-l-q-dcl-dcl-dcl-dcl-s-aa-aa-aa-aa-ay-ay-ay-n-n-n-jh-z-s-z-v-ix-ng-ng-n-h#-kcl')
    print(pred)
    
    # then convert to sound with:
    # https://www.0n0e.com/public/phoneme-synthesis/
    # or locally with `espeak -v en-US --ipa "[[ ... ]]"`
