CREATE DATABASE IF NOT EXISTS optuna_pmod;
CREATE USER 'pmod'@'%' IDENTIFIED BY 'pmod';
GRANT ALL PRIVILEGES ON optuna_pmod . * TO 'pmod'@'%';
FLUSH PRIVILEGES;
