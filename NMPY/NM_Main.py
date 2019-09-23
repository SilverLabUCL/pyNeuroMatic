# -*- coding: utf-8 -*-
"""
Main.py -  Defines NM Main class
Copyright 2019 Jason Rothman
"""


class Main:
    """NM Main class"""
    def __init__(self):
        self.experiments = []  # list of existing Experiments
        self.current_experiment = None  # select Experiment
        name = self.next_experiment_name()
        if name is not None:
            self.new_experiment(name, current=True);  # create default Experiment
        print("NM Main class created")
    def quotes(text):
        return "\"" + text + "\""
    def next_experiment_name(self):
        """
        Create next default name for an Experiment.
        
        Returns:
            name (str) if successful, None otherwise
        """
        n = 10 + len(self.experiments)
        for i in range(0,n):
            name = "NMExp" + str(i)
            found = False
            for e in self.experiments:
                if name.casefold() == e.name.casefold():
                    found = True;
                    break
            if not found:
                return name
        return None
    def check_experiment_name(self, name=None, exists=False, notexists=False):
        """
        Check if str name is OK to use for an Experiment.
        This function removes all special characters (except '_') before computing tests.
        
        Args:
            name (str): name to check
            exists (bool): check if name already exists as an Experiment
            notexists (bool): check if name does not exist as an Experiment
        
        Returns:
            name (str) if successful, None otherwise
        """
        if name is None:
            print("bad Experiment name: None")
            return None
        if len(name) == 0:
            print("bad Experiment name: 0 length")
            return None
        temp = ""
        for e in name:
            if e.isalnum() or e == "_":
                temp += e  # remove special characters
        name = temp
        if len(name) == 0:
            print("bad Experiment name: 0 length")
            return None
        if exists:
            for e in self.experiments:
                if name.casefold() == e.name.casefold():
                    print("Experiment " + Main.quotes(name) + " already exists")
                    return None
        if notexists:
            found = False;
            for e in self.experiments:
                if name.casefold() == e.name.casefold():
                    found = True
                    break
            if not found:
                print("Experiment " + Main.quotes(name) + " does not exist")
                return None
        return name  # name is OK
    def new_experiment(self, name=None, current=True):
        """
        Create a new Experiment.
        
        Args:
            name (str): name of new Experiment
            current (bool): set new Experiment as current Experiment
        
        Returns:
            Experiment (obj) if successful, None otherwise
        """
        if name is None or len(name) == 0:
            name = self.next_experiment_name()
        name = self.check_experiment_name(name=name, exists=True)
        if name is None:
            return None
        e = Experiment(name=name)
        self.experiments.append(e)
        print("created Experiment " + Main.quotes(name))
        if self.current_experiment is None or current:
            self.current_experiment = e
            print("set current Experiment " + Main.quotes(name))
        return e;
    def kill_experiment(self, name=None):
        """
        Kill an Experiment.
        
        Args:
            name (str): name of Experiment
        
        Returns:
            True for success, False otherwise
        """
        name = self.check_experiment_name(name=name, notexists=True)
        if name is None:
            return False
        kill = None
        for e in self.experiments:
            if name.casefold() == e.name.casefold():
                kill = e
                break
        if kill is not None:
            current = kill is self.current_experiment
            self.experiments.remove(kill)
            if current:
                if len(self.experiments) == 0:
                    self.current_experiment = None
                else:
                    self.current_experiment = self.experiments[0]
            print("killed Experiment " + Main.quotes(name))
            return True;
        return False
    def set_current_experiment(self, name=None):
        """
        Set the current Experiment.
        
        Args:
            name (str): name of Experiment
        
        Returns:
            True for success, False otherwise
        """
        name = self.check_experiment_name(name=name, notexists=True)
        if name is None:
            return False
        for e in self.experiments:
            if name.casefold() == e.name.casefold():
                self.current_experiment = e
                return True
        return False


class Experiment:
    """NM Experiment class"""
    def __init__(self, name=None):
        self.name = name
        
