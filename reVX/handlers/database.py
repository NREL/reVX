# -*- coding: utf-8 -*-
"""reVX SQL Database handler.

Created on Thu Aug 22 08:58:37 2019

@author: gbuster
"""
import pandas as pd
import psycopg2
import getpass
import time


class Database:
    """Framework to interact with the reV/reVX database."""

    def __init__(self, db_name, db_host='gds_edit.nrel.gov', db_user=None,
                 db_pass=None, db_port=5432):
        """
        Parameters
        ----------
        db_name : str
            Database name.
        db_host : str
            Database host name.
        db_user : str
            Your database user name.
        db_pass : str
            Database password (None if your password is cached).
        db_port : int
            Database port.
        """

        self.db_host = db_host or 'localhost'
        self.db_name = db_name or db_user or getpass.getuser()
        self.db_user = db_user or getpass.getuser()
        self.db_pass = db_pass
        self.db_port = db_port or 5432

        self._con = psycopg2.connect(host=self.db_host,
                                     port=self.db_port,
                                     user=self.db_user,
                                     password=self.db_pass,
                                     database=self.db_name)

    @property
    def con(self):
        """Return the database connection object."""
        return self._con

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """close DB connection on exit."""
        self._con.close()

    @classmethod
    def get_table(cls, table, schema, db_name, wait=300,
                  db_host='gds_edit.nrel.gov', db_user=None, db_pass=None,
                  db_port=5432):
        """Get a table using a database query.

        Parameters
        ----------
        table : str
            Table name to retrieve from schema.
        schema : str
            Schema name being accessed in the database.
        db_name : str
            Database name.
        wait : int
            Integer seconds to wait for DB connection to become available
            before raising exception.
        db_host : str
            Database host name.
        db_user : str
            Your database user name.
        db_pass : str
            Database password (None if your password is cached).
        db_port : int
            Database port.

        Returns
        -------
        df : pd.DataFrame
            Dataframe representation of schema.table.
        """
        sql = 'SELECT * FROM "{}"."{}"'.format(schema, table)
        kwargs = {"db_host": db_host, "db_user": db_user,
                  "db_pass": db_pass, "db_port": db_port}

        if wait > 0:
            waited = 0
            while True:
                try:
                    with cls(db_name, **kwargs) as db:
                        df = pd.read_sql(sql, db.con)
                except psycopg2.OperationalError as e:
                    if waited > wait:
                        raise e
                    else:
                        waited += 10
                        time.sleep(10)
                else:
                    break
        else:
            with cls(db_name, **kwargs) as db:
                df = pd.read_sql(sql, db.con)

        return df
