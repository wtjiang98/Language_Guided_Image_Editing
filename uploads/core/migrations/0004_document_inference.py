# Generated by Django 2.0.7 on 2020-03-29 21:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0003_auto_20200329_2130'),
    ]

    operations = [
        migrations.AddField(
            model_name='document',
            name='inference',
            field=models.FileField(null=True, upload_to='inference/'),
        ),
    ]
