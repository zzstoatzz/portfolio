import socket, random, time, os, sys, glob, re
import helpers as s
import config as c
import weebot

def main():
    # Setup
    cache = sys.argv[1]
    stoat = weebot.quoter(cache)
    stoat.load()
    stoat.connect()
    stoat.identify()
    stoat.join(c.channel)
    # Interaction
    stoat.converse()


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] != "cache":
        print("ERROR: A string representing the script directory must be provided. Please enter the exact name of the directory.\n")
        print("usage: python3 quote.py <dir>")
    else:
        try:
            main()
        except KeyboardInterrupt:
            print('And we out')
        except ConnectionResetError:
            print('Connection reset by peer -- ping error?')
