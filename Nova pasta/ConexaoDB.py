import mysql.connector

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="paulo123"
)

mysql_link = mydb.cursor()


def insertDB(paht, id, nome, tel, per, log, bai, uf, cep):
    sql = "INSERT INTO BEM_T.FACE (FACE_ID, LOC, NOME, TEL, PER, ATV) VALUES(%s,%s,%s,%s,%s,%s);"
    val = (id, paht, nome, tel, per, True)
    mysql_link.execute(sql, val, )

    sql1 = "INSERT INTO BEM_T.ENDE(ENDE_ID, LOG, BAI, UF, CEP) VALUES(%s,%s,%s,%s,%s);"
    val1 = (id, log, bai, uf, cep)
    mysql_link.execute(sql1, val1, )
    mydb.commit()
    return mysql_link.rowcount


def selectFaceDB(id):
    sql = "SELECT * FROM BEM_T.FACE,BEM_T.ENDE   WHERE FACE_ID = %s  AND  ATV=TRUE AND FACE_ID=ENDE_ID;"
    val = (id,)
    mysql_link.execute(sql, val)
    result = mysql_link.fetchall()
    return result


def updateDB(id, nome, tel, per, log, bai, uf, cep):
    sql = "UPDATE BEM_T.FACE SET `NOME` = %s, `TEL` = %s, `PER` = %s WHERE(`FACE_ID` = %s);"
    val = (nome, tel, per, id)
    mysql_link.execute(sql, val, )

    sql1 = "UPDATE BEM_T.ENDE SET `LOG` = %s, `BAI` = %s, `UF` = %s, `CEP` = %s WHERE (`ENDE_ID` = %s);"
    val1 = (log, bai, uf, cep, id)
    mysql_link.execute(sql1, val1, )
    mydb.commit()
    return mysql_link.rowcount


def deletDB(id, atv):
    sql = "UPDATE BEM_T.FACE SET `ATV` = %s WHERE (`FACE_ID` = %s);"
    val = (atv, id)
    mysql_link.execute(sql, val, )

    mydb.commit()
    return mysql_link.rowcount


def listaFaceDB():
    sql = "SELECT * FROM BEM_T.FACE  ORDER BY FACE_ID DESC LIMIT 10;"
    mysql_link.execute(sql)
    result = mysql_link.fetchall()
    return result
