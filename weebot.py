#!/usr/bin/env python3
import socket, random, time, glob
import config as c
import helpers as s
import info as i
import timeit

# title = HTML('Loading <style bg="yellow" fg="black">4 files...</style>')
# label = HTML('<ansired>some file</ansired>: ')

class quoter:
    def __init__(self, cache, servers=c.server):
        self.dir = cache + "/*.txt"
        self.cache = cache
        self.nick = 'StoatIncarnate'
        self.DOB = time.time()
        self.loadTime = 15 #sec
        self.age = 0 #seconds
        self.sock = socket.socket()
        self.listener = self.sock.makefile(mode='rw', buffering=1, encoding=c.ce, newline='\r\n')
        self.scripts = []
        self.quotes = []
        self.keywords = ['last']
        self.last = 'init string -- no one should ever see this'
        self.files = glob.glob(self.dir)
        self.status = self.status()
        self.status.serv = {}
        for s in servers: # add server and connected status dict
            self.status.serv[s] = False
        self.status.port = 6667
        self.status.acquainted = {"#stoattalk":False}
        self.status.muzzled = False
    class status:
        def isAcquainted(self,channel):
            return self.acquainted[channel]
    def connect(self, server = c.server, all=True):
        if all:
            for s in self.status.serv:
                self.sock.connect((s, self.status.port))
                self.status.serv[s] = True
        else:
            self.sock.connect((server, self.status.port))
            self.status.serv[server] = True
    def join(self, channel):
        j_str = "JOIN "+ channel
        if channel not in self.status.acquainted:
            self.status.acquainted[channel] = False
        print(j_str, file=self.listener)
    def identify(self):
        print('NICK', self.nick, file=self.listener)
        print('USER', self.nick, self.nick, self.nick, ':'+self.nick, file=self.listener)
        i_str = "NS IDENTIFY " + c.password +"\r\n"
        print(i_str, file = self.listener)
    def message(self, msg, chan, hold=False):
        self.sock.send(bytes("PRIVMSG "+chan+" :"+msg+"\n", c.ce))
        if hold:
            self.last = msg
    def greet(self, chan):
        i.greet(self, chan)
    def load(self):
        #s.mystify('arxiv/stoat.txt', self.DOB,self.loadTime)
            self.scripts = [s.vectorizetext(i) for i in self.files]
            self.files = [i.strip(self.cache).strip('.txt').strip('/') for i in self.files]
            for script in self.scripts:
                for quote in script:
                    self.quotes.append(quote.lower())
    def pong(self, line):
        print("PONG :" + line.split(':')[1], file=self.listener)
    def getQuote(self, line):
        line = line.lower().replace("!find", "")
        line = line.split(':')[-1].strip()
        qs = self.quotes.copy()
        quote = s.find(line, qs)
        if (quote == -1):
            msg = "Input was not detailed enough to isolate a quote. Please include more of the quote."
            self.message(msg, c.channel)
            return
        elif (isinstance(quote, list)):
            err = "Couldn't isolate quote. It might be one of these...\n"
            self.message(err, c.channel)
            if len(quote) > 5:
                quote = quote[0:5]
            for q in quote:
                self.message("-    "+q, c.channel)
            return
        i = 0
        for script in self.scripts:
            for q in script:
                if q.lower() == quote:
                    self.message(q, c.channel, True)
                    return q
        self.message(msg, c.channel)
    def follow(self, line, rev):
        line = line.lower().replace("!follow", "")
        line = line.split(':')[-1].strip()
        qs = self.quotes.copy()
        quote = s.find(line, qs)
        if (quote == -1):
            msg = "Input was not detailed enough to isolate a quote. Please include more of the quote."
            self.message(msg, c.channel)
        elif (isinstance(quote, list)):
            #err = "Which quote were you referring to? Indicate by sending the index (starting at 0) preceeded by the '!' operator (e.g. '!3')\n"
            #self.message(err, c.channel)
            for q in quote:
                self.message(q, c.channel, True)
            return quote
        for script in self.scripts:
            for q in script:
                if q.lower() == quote:
                    j = script.index(q)
                    p = script[j+1]
                    if not rev:
                        self.message(script[j+1], c.channel)
                    else:
                        self.message(script[j-1], c.channel)

                    return p
    def sample(self, line):
        line = line.replace("!sample", "").strip()
        for i in range(0, len(self.files)):
            if self.files[i] in line:
                q = random.choice(self.scripts[i])
                self.message(q, c.channel, True)
                return q
        msg = "Input did not match any existing script names. Please try again."
        self.message(msg, c.channel)
    def info(self):
        i.info(self)
    def choose(self, line):
        r = ''
        if "!info" in line:
            self.info()
        elif "!find" in line:
            r = self.getQuote(line)
        elif "!follow" in line:
            r = self.follow(line, False)
        elif "!prev" in line:
            r = self.follow(line, True)
        elif "!sample" in line:
            r = self.sample(line)
        elif "!quiet" in line:
            self.status.muzzled = True
        elif "!respawn" in line:
            self.respawn()
        if r != '':
            return r
    def converse(self):
        wantLast = False
        for line in self.listener:
            self.age = time.time() - self.DOB
            line = line.strip()
            if "PING" in line:
                self.pong(line)
            elif "!" in line:
                line, wantLast = self.heed(line) # handle special commands
                if not wantLast:
                    if self.age > 15:
                        if (not self.status.isAcquainted(c.channel)):
                            self.greet(c.channel)
                            self.status.acquainted[c.channel] = True
                    if not self.status.muzzled:
                        self.choose(line)
    def heed(self, line):
        b = False
        if "!die" in line:
            self.die()
        elif "!listen" in line:
            if not self.status.muzzled:
                self.message("I was listening already, asshole...", c.channel)
            self.status.muzzled = False
        elif ".last" in line:
            self.choose(self.last)
            b = True
        return line, b
    def die(self):
        self.message("Yeah fuck off anyways", c.channel)
        print("Someone killed StoatIncarnate")
        exit()
    def respawn(self):
        self.__init__(self.cache, self.status.serv)
